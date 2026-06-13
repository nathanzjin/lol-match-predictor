#!/usr/bin/env python3
"""
predict_v3.py - predict a matchup from each team's MOST RECENT ROSTER.

Where predict.py used team rolling form, this uses the player-level model from
train_v3.py. The flow is exactly the thing the player work was about:

  1. Roster association - look up the five players each team most recently
     fielded in each role (player_features.most_recent_roster), so the prediction
     is about the lineups that will actually take the stage, not a stale team.
  2. Player signals - mean player-Elo per side (ratings travel with players),
     each player's recent per-role box-score form, and player experience.
  3. Team context - the region-anchored team rating, folded in.
  4. Feed the blue-minus-red differentials to the saved v3 pipeline.

Because a hypothetical matchup has no "previous game" to diff against, roster
continuity is assumed full (5/5) - i.e. these are the teams' settled lineups.
Override a roster on the command line to explore a sub or a transfer.

Examples:
  python predict_v3.py "T1" "Gen.G"
  python predict_v3.py "T1" "Gen.G" --window 15
  python predict_v3.py "T1" "Gen.G" --blue-roster top=Zeus
  python predict_v3.py --list
  python predict_v3.py --roster "T1"
"""
from __future__ import annotations
import argparse
import difflib
import sys

import joblib
import numpy as np
import pandas as pd

from train_v1 import MODEL_DIR
from player_features import (
    ROLES, PLAYER_STATS, MAJOR_REGIONS, load_player_rows, add_player_rolling,
    most_recent_roster, team_home_region,
)


def resolve(name: str, available: list[str], all_teams: list[str]) -> str:
    if name in available:
        return name
    lower = {t.lower(): t for t in available}
    if name.lower() in lower:
        return lower[name.lower()]
    # Named team exists but isn't in a supported Tier-1 region
    all_lower = {t.lower(): t for t in all_teams}
    if name in all_teams or name.lower() in all_lower:
        real = name if name in all_teams else all_lower[name.lower()]
        sys.exit(f"'{real}' is outside the supported Tier-1 regions "
                 f"({', '.join(MAJOR_REGIONS)}). Only Tier-1 teams are supported for "
                 f"prediction; minor-region data is used for training only.")
    close = difflib.get_close_matches(name, available, n=5, cutoff=0.6)
    msg = f"Team '{name}' not found."
    if close:
        msg += " Did you mean: " + ", ".join(map(repr, close)) + "?"
    sys.exit(msg + "\nUse --list to see supported team names.")


def player_recent_form(rolled: pd.DataFrame, player: str, window: int) -> tuple[dict, int]:
    """Mean of a player's last `window` games for each stat + their game count.

    Uses the raw stat columns (not the shifted rolling ones): for a *future*
    game we want the player's latest form, so the most recent games are included.
    """
    sub = rolled[rolled["playername"] == player].sort_values(["date", "gameid"])
    tail = sub.tail(window)
    form = {s: float(tail[s].mean()) if len(tail) else np.nan for s in PLAYER_STATS}
    return form, int(len(sub))


def parse_roster_overrides(items: list[str] | None) -> dict[str, str]:
    out: dict[str, str] = {}
    for it in items or []:
        if "=" not in it:
            sys.exit(f"Bad --*-roster entry '{it}'. Use role=Player, e.g. top=Zeus.")
        role, name = it.split("=", 1)
        role = role.strip().lower()
        if role not in ROLES:
            sys.exit(f"Unknown role '{role}'. Roles: {', '.join(ROLES)}.")
        out[role] = name.strip()
    return out


def side_features(team, roster, rolled, state, region_of, window):
    """Per-side raw pieces: role->form, role->career, plain & anchored mean
    player-Elo, region-anchored team rating, region."""
    pbase = state["pelo_params"]["base"]
    ratings = state["player_ratings"]
    ra_player, ra_region = state["player_ra_player"], state["player_ra_region"]
    rap = state["pelo_ra_params"]
    rel = state["relo_params"]
    region = region_of.get(team, "OTHER")

    forms, careers, elos, elos_ra = {}, {}, [], []
    for role in ROLES:
        p = roster[role]
        f, n = player_recent_form(rolled, p, window)
        forms[role], careers[role] = f, n
        elos.append(ratings.get(p, pbase))
        # anchored effective = within-region player skill + region strength
        eff = ra_player.get(p, rap["base"]) + rap["beta"] * (ra_region.get(region, rap["base"]) - rap["base"])
        elos_ra.append(eff)
    mean_elo = float(np.mean(elos))
    mean_elo_ra = float(np.mean(elos_ra))

    team_r = state["region_team"].get(team, rel["base"])
    region_r = state["region_region"].get(region, rel["base"])
    effective = team_r + rel["beta"] * (region_r - rel["base"])
    return forms, careers, mean_elo, mean_elo_ra, effective, region


def main() -> None:
    ap = argparse.ArgumentParser(description="Predict a LoL matchup from recent rosters (player-level v3 model).")
    ap.add_argument("blue", nargs="?", help="Blue-side team")
    ap.add_argument("red", nargs="?", help="Red-side team")
    ap.add_argument("--window", type=int, default=10, help="Games of recent form per player (default 10)")
    ap.add_argument("--blue-roster", nargs="+", help="Override blue players, e.g. top=Zeus mid=Faker")
    ap.add_argument("--red-roster", nargs="+", help="Override red players")
    ap.add_argument("--list", action="store_true", help="List supported Tier-1 teams by region and exit")
    ap.add_argument("--roster", metavar="TEAM", help="Print a team's most-recent roster and exit")
    args = ap.parse_args()

    pipe_path = MODEL_DIR / "lol_pipeline_v3.joblib"
    state_path = MODEL_DIR / "rating_state_v3.joblib"
    if not pipe_path.exists() or not state_path.exists():
        sys.exit("v3 artifacts missing. Train first:  python train_v3.py")

    players = load_player_rows()
    rolled = players  # raw stats per player row; recent form taken as tail-mean
    all_teams = sorted(players["teamname"].unique())
    region_of = team_home_region(players)
    # Supported teams: only those in a Tier-1 region (minor regions are
    # training-breadth only, never a prediction target).
    available = sorted(t for t in all_teams if region_of.get(t) in MAJOR_REGIONS)

    if args.list:
        by_region: dict[str, list[str]] = {r: [] for r in MAJOR_REGIONS}
        for t in available:
            by_region[region_of[t]].append(t)
        print(f"Supported Tier-1 teams ({len(available)}), by region:")
        for r in MAJOR_REGIONS:
            print(f"\n[{r}] ({len(by_region[r])})")
            print("  " + "\n  ".join(by_region[r]))
        return
    if args.roster:
        team = resolve(args.roster, available, all_teams)
        r = most_recent_roster(players, team)
        print(f"{team} most-recent roster ({region_of.get(team)}):")
        for role in ROLES:
            print(f"  {role}: {r.get(role, '(unknown)')}")
        return
    if not args.blue or not args.red:
        sys.exit('Provide two teams:  python predict_v3.py "<blue>" "<red>"  (or --list)')

    blue = resolve(args.blue, available, all_teams)
    red = resolve(args.red, available, all_teams)
    if blue == red:
        sys.exit("Blue and red must be different teams.")

    pipe = joblib.load(pipe_path)
    features = joblib.load(MODEL_DIR / "feature_cols_v3.joblib")
    state = joblib.load(state_path)
    ratings = state["player_ratings"]

    blue_roster = most_recent_roster(players, blue)
    red_roster = most_recent_roster(players, red)
    blue_roster.update(parse_roster_overrides(args.blue_roster))
    red_roster.update(parse_roster_overrides(args.red_roster))
    for tag, roster in [(blue, blue_roster), (red, red_roster)]:
        missing = [r for r in ROLES if r not in roster]
        if missing:
            sys.exit(f"Incomplete roster for {tag}: missing {missing}. "
                     f"Provide with --{'blue' if tag == blue else 'red'}-roster role=Player.")

    bf, bc, b_elo, b_elo_ra, b_eff, b_reg = side_features(blue, blue_roster, rolled, state, region_of, args.window)
    rf, rc, r_elo, r_elo_ra, r_eff, r_reg = side_features(red, red_roster, rolled, state, region_of, args.window)

    # Assemble the exact training feature row
    row: dict[str, float] = {}
    for role in ROLES:
        for s in PLAYER_STATS:
            row[f"diff_{role}_{s}"] = bf[role][s] - rf[role][s]
        row[f"diff_{role}_career"] = bc[role] - rc[role]
    row["blue_continuity"] = 5.0      # hypothetical settled lineups
    row["red_continuity"] = 5.0
    row["min_continuity"] = 5.0
    row["pelo_diff"] = b_elo - r_elo
    row["pelo_ra_diff"] = b_elo_ra - r_elo_ra
    row["relo_diff"] = b_eff - r_eff

    X = pd.DataFrame([row])[features]
    p_blue = float(pipe.predict_proba(X)[:, 1][0])
    p_red = 1.0 - p_blue
    winner, conf = (blue, p_blue) if p_blue >= 0.5 else (red, p_red)

    def show_roster(tag, roster, careers, mean_elo, mean_elo_ra, region):
        print(f"  {tag}  (region {region}, mean player-Elo {mean_elo:.0f}, anchored {mean_elo_ra:.0f})")
        for role in ROLES:
            p = roster[role]
            print(f"    {role}: {p:<16} elo={ratings.get(p, state['pelo_params']['base']):6.0f}  games={careers[role]}")

    print(f"\nMatchup (player-level v3 model)")
    print("BLUE")
    show_roster(blue, blue_roster, bc, b_elo, b_elo_ra, b_reg)
    print("RED")
    show_roster(red, red_roster, rc, r_elo, r_elo_ra, r_reg)

    print("\nKey differentials (blue - red):")
    print(f"  player-Elo (mean)        {row['pelo_diff']:+.1f}")
    print(f"  player-Elo (anchored)    {row['pelo_ra_diff']:+.1f}")
    print(f"  region-anchored team     {row['relo_diff']:+.1f}")
    for role in ROLES:
        print(f"  {role} dpm form            {row[f'diff_{role}_dpm']:+.1f}")

    print("\nPrediction:")
    print(f"  P({blue} wins, blue side) = {p_blue:.1%}")
    print(f"  P({red} wins, red side)   = {p_red:.1%}")
    print(f"  -> {winner} favored ({conf:.1%})")
    print("\nNote: blue side carries a ~53% base-rate edge; swapping sides shifts the number.")


if __name__ == "__main__":
    main()
