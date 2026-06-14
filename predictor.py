#!/usr/bin/env python3
"""
predictor.py - reusable core for the player-level match predictor.

Both the CLI (predict.py) and the API (api.py) use this. It loads the trained
artifacts and player data once, then serves three things:

  * teams_by_region()        - supported Tier-1 teams grouped by region
  * roster(team)             - a team's most-recent roster (per role)
  * predict(blue, red, ...)  - a structured prediction for a hypothetical matchup

The prediction rebuilds the exact feature row train_v3.py used: per-role recent
form differentials, plain and region-anchored mean player-Elo, and the region-
anchored team rating, fed to the saved XGBoost pipeline. Roster continuity is
assumed full (5/5) for a hypothetical settled lineup.

Errors are raised as PredictError (callers decide how to surface them) rather
than calling sys.exit, so the API can return clean 400s.
"""
from __future__ import annotations
import difflib

import joblib
import numpy as np
import pandas as pd

from train_v1 import MODEL_DIR
from player_features import (
    ROLES, PLAYER_STATS, MAJOR_REGIONS, load_player_rows, most_recent_roster,
    team_home_region,
)

PIPE_PATH = MODEL_DIR / "lol_pipeline_v3.joblib"
STATE_PATH = MODEL_DIR / "rating_state_v3.joblib"
FEATURES_PATH = MODEL_DIR / "feature_cols_v3.joblib"


class PredictError(ValueError):
    """Bad prediction request (unknown/unsupported team, bad roster, etc.)."""


def player_recent_form(rolled: pd.DataFrame, player: str, window: int) -> tuple[dict, int]:
    """Mean of a player's last `window` games per stat + their game count.

    Uses raw stat columns (not the shifted rolling ones): for a *future* game we
    want the player's latest form, so the most recent games are included.
    """
    sub = rolled[rolled["playername"] == player].sort_values(["date", "gameid"])
    tail = sub.tail(window)
    form = {s: float(tail[s].mean()) if len(tail) else float("nan") for s in PLAYER_STATS}
    return form, int(len(sub))


class Predictor:
    def __init__(self) -> None:
        for p in (PIPE_PATH, STATE_PATH, FEATURES_PATH):
            if not p.exists():
                raise FileNotFoundError(
                    f"Model artifact missing: {p}. Train first:  python train_v3.py")
        self.players = load_player_rows()
        self.region_of = team_home_region(self.players)
        self.all_teams = sorted(self.players["teamname"].unique())
        self.available = sorted(t for t in self.all_teams
                                if self.region_of.get(t) in MAJOR_REGIONS)
        self.pipe = joblib.load(PIPE_PATH)
        self.features = joblib.load(FEATURES_PATH)
        self.state = joblib.load(STATE_PATH)

    # ---- catalog -------------------------------------------------------- #
    def teams_by_region(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {r: [] for r in MAJOR_REGIONS}
        for t in self.available:
            out[self.region_of[t]].append(t)
        return out

    def roster(self, team: str) -> dict[str, str]:
        return most_recent_roster(self.players, team)

    def resolve(self, name: str) -> str:
        """Map a (possibly imperfect) name to a supported Tier-1 team or raise."""
        if name in self.available:
            return name
        lower = {t.lower(): t for t in self.available}
        if name.lower() in lower:
            return lower[name.lower()]
        all_lower = {t.lower(): t for t in self.all_teams}
        if name in self.all_teams or name.lower() in all_lower:
            real = name if name in self.all_teams else all_lower[name.lower()]
            raise PredictError(
                f"'{real}' is outside the supported Tier-1 regions "
                f"({', '.join(MAJOR_REGIONS)}). Minor-region teams are training "
                f"breadth only, not a prediction target.")
        close = difflib.get_close_matches(name, self.available, n=5, cutoff=0.6)
        hint = (" Did you mean: " + ", ".join(map(repr, close)) + "?") if close else ""
        raise PredictError(f"Team '{name}' not found.{hint}")

    # ---- per-side feature assembly -------------------------------------- #
    def _side(self, team: str, roster: dict[str, str], window: int) -> dict:
        st = self.state
        pbase = st["pelo_params"]["base"]
        ratings = st["player_ratings"]
        ra_player, ra_region = st["player_ra_player"], st["player_ra_region"]
        rap, rel = st["pelo_ra_params"], st["relo_params"]
        region = self.region_of.get(team, "OTHER")

        forms, players_out, elos, elos_ra = {}, [], [], []
        for role in ROLES:
            p = roster[role]
            f, n = player_recent_form(self.players, p, window)
            forms[role] = f
            elo = ratings.get(p, pbase)
            elos.append(elo)
            eff = ra_player.get(p, rap["base"]) + rap["beta"] * (
                ra_region.get(region, rap["base"]) - rap["base"])
            elos_ra.append(eff)
            players_out.append({"role": role, "player": p,
                                "elo": round(elo, 1), "games": n})

        team_r = st["region_team"].get(team, rel["base"])
        region_r = st["region_region"].get(region, rel["base"])
        effective = team_r + rel["beta"] * (region_r - rel["base"])
        return {
            "team": team, "region": region,
            "mean_elo": float(np.mean(elos)),
            "anchored_elo": float(np.mean(elos_ra)),
            "team_effective": float(effective),
            "roster": players_out, "_forms": forms,
        }

    # ---- prediction ----------------------------------------------------- #
    def predict(self, blue: str, red: str, window: int = 10,
                blue_overrides: dict[str, str] | None = None,
                red_overrides: dict[str, str] | None = None) -> dict:
        blue = self.resolve(blue)
        red = self.resolve(red)
        if blue == red:
            raise PredictError("Blue and red must be different teams.")

        b_roster = self.roster(blue)
        r_roster = self.roster(red)
        b_roster.update(blue_overrides or {})
        r_roster.update(red_overrides or {})
        for tag, roster in [(blue, b_roster), (red, r_roster)]:
            missing = [r for r in ROLES if r not in roster or not roster[r]]
            if missing:
                raise PredictError(f"Incomplete roster for {tag}: missing {missing}.")

        B = self._side(blue, b_roster, window)
        R = self._side(red, r_roster, window)

        row: dict[str, float] = {}
        role_dpm = {}
        for role in ROLES:
            for s in PLAYER_STATS:
                row[f"diff_{role}_{s}"] = B["_forms"][role][s] - R["_forms"][role][s]
            row[f"diff_{role}_career"] = (
                next(p["games"] for p in B["roster"] if p["role"] == role)
                - next(p["games"] for p in R["roster"] if p["role"] == role))
            role_dpm[role] = round(row[f"diff_{role}_dpm"], 1)
        row["blue_continuity"] = 5.0
        row["red_continuity"] = 5.0
        row["min_continuity"] = 5.0
        row["pelo_diff"] = B["mean_elo"] - R["mean_elo"]
        row["pelo_ra_diff"] = B["anchored_elo"] - R["anchored_elo"]
        row["relo_diff"] = B["team_effective"] - R["team_effective"]

        X = pd.DataFrame([row])[self.features]
        p_blue = float(self.pipe.predict_proba(X)[:, 1][0])
        winner = blue if p_blue >= 0.5 else red
        for side in (B, R):
            side.pop("_forms", None)
        return {
            "blue": B, "red": R,
            "diffs": {
                "pelo_diff": round(row["pelo_diff"], 1),
                "pelo_ra_diff": round(row["pelo_ra_diff"], 1),
                "relo_diff": round(row["relo_diff"], 1),
                "role_dpm": role_dpm,
            },
            "p_blue": p_blue,
            "p_red": 1.0 - p_blue,
            "winner": winner,
            "confidence": max(p_blue, 1.0 - p_blue),
            "window": window,
        }
