#!/usr/bin/env python3
"""
player_features.py - individual-player signals for LoL match prediction.

Everything the project has done so far rates *teams* (rolling team form, team
Elo, region-anchored Elo). But a team is just its five players, and rosters
move: a star mid transfers, an academy sub fills in for a week, two orgs swap
bot lanes in the off-season. Team-level signals are blind to this until enough
new games accumulate; player-level signals update the instant the lineup changes.

This module turns the per-player rows in the Oracle's Elixir data into three
kinds of pre-game signal, all leakage-safe (every number uses only games that
finished strictly before the game being predicted):

  1. Player Elo that TRAVELS WITH THE PLAYER (`PlayerElo`).
     One rating per `playername`. A team's strength is the mean of its five
     players' ratings, so when a player changes orgs their skill goes with
     them and the new team is rated correctly from game one. Updated on the
     shared team result (all five move together).

  2. Per-role rolling individual stats (`add_player_rolling`).
     For each player, the trailing-window mean of metrics that exist for BOTH
     'complete' and 'partial' rows - dpm, cspm, vspm, earned gpm, damageshare,
     kda - so LPL (always 'partial') and international games are covered, unlike
     the v1 form features which lean on golddiffat15 (null for partial rows).
     These become per-role differentials (e.g. diff_mid_dpm = blue mid - red mid).

  3. Roster continuity (`roster_continuity`).
     How many of a team's five players also started its previous game. Low
     continuity flags a lineup the team-level signals haven't caught up with -
     exactly when the player-level view should help most.

Roster association: `most_recent_roster(team, as_of)` returns the five players a
team most recently fielded in each role before a date - the "current roster" for
predicting a hypothetical matchup. The backtest instead uses each game's actual
lineup (known pre-game in reality), scored from each player's prior games only.

Identity: keyed on `playername` (present ~99.99% of rows; `playerid` is null ~1%
of the time and ~120 players carry more than one id).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path("data/raw")
YEARS = [2023, 2024, 2025, 2026]
ROLES = ["top", "jng", "mid", "bot", "sup"]

# Stats present for BOTH complete and partial rows (keeps LPL + international in
# scope). golddiffat15/xpdiffat15 are deliberately excluded: null for partials.
PLAYER_STATS = ["dpm", "cspm", "vspm", "earned gpm", "damageshare", "kda"]

READ_COLS = [
    "gameid", "datacompleteness", "league", "date", "side", "position",
    "playername", "teamname", "result", "gamelength",
    "kills", "deaths", "assists", "dpm", "cspm", "vspm", "earned gpm",
    "damageshare",
]


# --------------------------------------------------------------------------- #
# Load player rows (partials kept, so LPL / international players are included)
# --------------------------------------------------------------------------- #
def load_player_rows(years: list[int] | None = None) -> pd.DataFrame:
    years = years or YEARS
    frames = [
        pd.read_csv(DATA_DIR / f"{y}_LoL_esports_match_data_from_OraclesElixir.csv",
                    usecols=READ_COLS, low_memory=False)
        for y in years
    ]
    df = pd.concat(frames, ignore_index=True)
    df = df[df["position"].isin(ROLES)]
    df = df[df["result"].isin([0, 1])]
    df = df[(df["gamelength"].isna()) | (df["gamelength"] > 900)]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "gameid", "teamname", "playername", "side", "position"])
    # KDA: (kills + assists) / max(deaths, 1) - a classic individual box-score stat
    df["kda"] = (df["kills"] + df["assists"]) / df["deaths"].clip(lower=1)
    # Keep team-games that have all five roles present (clean lineups only)
    full = df.groupby(["gameid", "teamname"])["position"].transform("nunique").eq(5)
    df = df[full]
    return df.sort_values(["date", "gameid"]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 1. Player Elo that travels with the player
# --------------------------------------------------------------------------- #
@dataclass
class PlayerElo:
    """Skill rating per player; team strength = mean of its five players.

    All five players on a side share the game result, so each is nudged by the
    same k*(result - p). Because the rating is keyed on the player (not the
    team), it follows the player through transfers and substitutions - the
    point of going player-level.
    """
    base: float = 1500.0
    k: float = 24.0
    home_adv: float = 20.0
    scale: float = 400.0
    ratings: dict[str, float] = field(default_factory=dict)

    def rating(self, player: str) -> float:
        return self.ratings.get(player, self.base)

    def team_strength(self, players: list[str]) -> float:
        return float(np.mean([self.rating(p) for p in players]))

    def expect_blue(self, blue: list[str], red: list[str]) -> float:
        diff = (self.team_strength(blue) + self.home_adv) - self.team_strength(red)
        return 1.0 / (1.0 + 10.0 ** (-diff / self.scale))

    def update(self, blue: list[str], red: list[str], blue_won: float) -> float:
        """Predict pre-game, then move every player's rating toward the result."""
        p = self.expect_blue(blue, red)
        delta = self.k * (float(blue_won) - p)
        for pl in blue:
            self.ratings[pl] = self.rating(pl) + delta
        for pl in red:
            self.ratings[pl] = self.rating(pl) - delta
        return p


def build_lineups(players: pd.DataFrame) -> pd.DataFrame:
    """One row per (gameid, team): side, result, role->player columns, date.

    This is the per-game lineup table everything else is built on.
    """
    idx = ["gameid", "teamname"]
    roster = (players.pivot_table(index=idx, columns="position",
                                  values="playername", aggfunc="first")
              .reset_index())
    meta = (players.drop_duplicates(idx)[idx + ["date", "side", "league", "result"]])
    out = meta.merge(roster, on=idx, how="inner")
    out = out.dropna(subset=ROLES)
    return out.sort_values(["date", "gameid"]).reset_index(drop=True)


def player_elo_stream(lineups: pd.DataFrame, **elo_kw) -> pd.DataFrame:
    """Replay games in order; record pre-game player-Elo strength per side.

    Returns one row per game with blue/red mean player rating and the model's
    pre-game blue-win probability. Leakage-free: predict, then update.
    """
    model = PlayerElo(**elo_kw)
    games = _pair_sides(lineups)
    b_str, r_str, probs = [], [], []
    for _, g in games.iterrows():
        blue = [g[f"blue_{r}"] for r in ROLES]
        red = [g[f"red_{r}"] for r in ROLES]
        b_str.append(model.team_strength(blue))
        r_str.append(model.team_strength(red))
        probs.append(model.update(blue, red, float(g["target"])))
    out = games.assign(blue_pelo=b_str, red_pelo=r_str,
                       pelo_diff=np.array(b_str) - np.array(r_str),
                       pelo_p=probs)
    out.attrs["model"] = model
    return out


# --------------------------------------------------------------------------- #
# 2. Per-role rolling individual stats (leakage-safe)
# --------------------------------------------------------------------------- #
def add_player_rolling(players: pd.DataFrame, window: int = 10,
                       min_games: int = 3) -> pd.DataFrame:
    """For each player, trailing-window mean of each stat, current game excluded.

    shift(1) before rolling => the current game never informs its own features.
    Also tags career game count (experience) up to but excluding this game.
    """
    df = players.sort_values(["date", "gameid"]).reset_index(drop=True)
    grp = df.groupby("playername", sort=False)
    for stat in PLAYER_STATS:
        df[f"roll_{stat}"] = grp[stat].transform(
            lambda s: s.shift(1).rolling(window, min_periods=min_games).mean()
        )
    df["career_games"] = grp.cumcount()   # games strictly before this one
    return df


def _pair_sides(lineups: pd.DataFrame, extra_cols: list[str] | None = None) -> pd.DataFrame:
    """Pivot the per-team lineup table to one row per game (blue vs red)."""
    extra_cols = extra_cols or []
    keep_blue = ["gameid", "date", "league", "result"] + ROLES + extra_cols
    keep_red = ["gameid"] + ROLES + extra_cols
    good = lineups.groupby("gameid")["side"].transform("nunique").eq(2)
    lu = lineups[good]
    blue = (lu[lu["side"] == "Blue"][keep_blue]
            .rename(columns={"result": "blue_result",
                             **{c: f"blue_{c}" for c in ROLES + extra_cols}}))
    red = (lu[lu["side"] == "Red"][keep_red]
           .rename(columns={c: f"red_{c}" for c in ROLES + extra_cols}))
    g = blue.merge(red, on="gameid", how="inner", suffixes=("", "_r"))
    g["target"] = g["blue_result"].astype(int)
    return g.sort_values(["date", "gameid"]).reset_index(drop=True)


def build_role_stat_lineups(players_rolled: pd.DataFrame) -> pd.DataFrame:
    """Per (gameid, team) row carrying each role-player's rolling stats.

    Output columns include, per role r and stat s, `r_roll_s` (e.g. mid_roll_dpm)
    plus per-role career_games, ready to be paired into blue-red role diffs.
    """
    idx = ["gameid", "teamname"]
    roll_cols = [f"roll_{s}" for s in PLAYER_STATS] + ["career_games"]
    # one player per (game, team, role); pivot stats into role-prefixed columns
    pieces = []
    meta = players_rolled.drop_duplicates(idx)[idx + ["date", "side", "league", "result"]]
    base = meta.set_index(idx)
    for role in ROLES:
        sub = (players_rolled[players_rolled["position"] == role]
               .drop_duplicates(idx)
               .set_index(idx)[roll_cols + ["playername"]]
               .rename(columns={**{c: f"{role}_{c}" for c in roll_cols},
                                "playername": role}))
        pieces.append(sub)
    out = base.join(pieces, how="left").reset_index()
    return out.sort_values(["date", "gameid"]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 3. Roster continuity
# --------------------------------------------------------------------------- #
def add_roster_continuity(lineups: pd.DataFrame) -> pd.DataFrame:
    """How many of a team's five players also started its previous game (0-5).

    Computed per team in time order, so it only ever looks backward.
    """
    lu = lineups.sort_values(["teamname", "date", "gameid"]).copy()
    cont = np.full(len(lu), np.nan)
    prev_by_team: dict[str, set] = {}
    for i, row in enumerate(lu.itertuples(index=False)):
        team = row.teamname
        cur = {getattr(row, r) for r in ROLES}
        prev = prev_by_team.get(team)
        if prev is not None:
            cont[i] = len(cur & prev)
        prev_by_team[team] = cur
    lu["roster_continuity"] = cont
    return lu.sort_values(["date", "gameid"]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Roster association: the "most recent roster" for a hypothetical matchup
# --------------------------------------------------------------------------- #
def most_recent_roster(players: pd.DataFrame, team: str,
                       as_of: pd.Timestamp | None = None) -> dict[str, str]:
    """The five players a team most recently fielded in each role before `as_of`.

    This is the roster-association tool for predicting a hypothetical game where
    the lineup isn't given: take the latest appearance in each role. Defaults to
    using all available history when `as_of` is None.
    """
    sub = players[players["teamname"] == team]
    if as_of is not None:
        sub = sub[sub["date"] < as_of]
    roster: dict[str, str] = {}
    for role in ROLES:
        r = sub[sub["position"] == role].sort_values(["date", "gameid"])
        if len(r):
            roster[role] = r["playername"].iloc[-1]
    return roster


if __name__ == "__main__":
    # Smoke test: load, show coverage, and print a couple of current rosters.
    players = load_player_rows()
    print(f"[player rows in scope] {len(players):,}")
    print(f"[team-games] {players.groupby(['gameid', 'teamname']).ngroups:,}")
    lineups = build_lineups(players)
    print(f"[clean lineups] {len(lineups):,}")
    for team in ["T1", "Gen.G", "JD Gaming", "G2 Esports"]:
        r = most_recent_roster(players, team)
        if r:
            print(f"  {team:<14} " + "  ".join(f"{k}:{v}" for k, v in r.items()))
