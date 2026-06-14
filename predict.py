#!/usr/bin/env python3
"""
predict.py - CLI to predict a matchup from each team's MOST RECENT ROSTER.

Thin command-line wrapper over predictor.Predictor (the same core the API uses).
It runs the player-level model trained by train_v3.py: roster association +
player-Elo (plain & region-anchored) + per-role recent form + region rating.

Predictions are supported for Riot Tier-1 region teams only (LCK/LPL/LEC/LCS/
CBLOL/LCP); minor-region teams are training breadth and are rejected here.

Examples:
  python predict.py "T1" "Gen.G"
  python predict.py "T1" "Gen.G" --window 15
  python predict.py "T1" "Gen.G" --blue-roster top=Zeus
  python predict.py --list
  python predict.py --roster "T1"
"""
from __future__ import annotations
import argparse
import sys

from player_features import ROLES
from predictor import Predictor, PredictError


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


def main() -> None:
    ap = argparse.ArgumentParser(description="Predict a LoL matchup from recent rosters (player-level model).")
    ap.add_argument("blue", nargs="?", help="Blue-side team")
    ap.add_argument("red", nargs="?", help="Red-side team")
    ap.add_argument("--window", type=int, default=10, help="Games of recent form per player (default 10)")
    ap.add_argument("--blue-roster", nargs="+", help="Override blue players, e.g. top=Zeus mid=Faker")
    ap.add_argument("--red-roster", nargs="+", help="Override red players")
    ap.add_argument("--list", action="store_true", help="List supported Tier-1 teams by region and exit")
    ap.add_argument("--roster", metavar="TEAM", help="Print a team's most-recent roster and exit")
    args = ap.parse_args()

    try:
        predictor = Predictor()
    except FileNotFoundError as e:
        sys.exit(str(e))

    if args.list:
        by_region = predictor.teams_by_region()
        total = sum(len(v) for v in by_region.values())
        print(f"Supported Tier-1 teams ({total}), by region:")
        for r, teams in by_region.items():
            print(f"\n[{r}] ({len(teams)})")
            print("  " + "\n  ".join(teams))
        return

    if args.roster:
        try:
            team = predictor.resolve(args.roster)
        except PredictError as e:
            sys.exit(str(e))
        r = predictor.roster(team)
        print(f"{team} most-recent roster ({predictor.region_of.get(team)}):")
        for role in ROLES:
            print(f"  {role}: {r.get(role, '(unknown)')}")
        return

    if not args.blue or not args.red:
        sys.exit('Provide two teams:  python predict.py "<blue>" "<red>"  (or --list)')

    try:
        res = predictor.predict(
            args.blue, args.red, window=args.window,
            blue_overrides=parse_roster_overrides(args.blue_roster),
            red_overrides=parse_roster_overrides(args.red_roster))
    except PredictError as e:
        sys.exit(str(e))

    blue, red = res["blue"], res["red"]

    def show(side):
        print(f"  {side['team']}  (region {side['region']}, mean player-Elo "
              f"{side['mean_elo']:.0f}, anchored {side['anchored_elo']:.0f})")
        for p in side["roster"]:
            print(f"    {p['role']}: {p['player']:<16} elo={p['elo']:6.0f}  games={p['games']}")

    print("\nMatchup (player-level model)")
    print("BLUE"); show(blue)
    print("RED"); show(red)

    d = res["diffs"]
    print("\nKey differentials (blue - red):")
    print(f"  player-Elo (mean)        {d['pelo_diff']:+.1f}")
    print(f"  player-Elo (anchored)    {d['pelo_ra_diff']:+.1f}")
    print(f"  region-anchored team     {d['relo_diff']:+.1f}")
    for role in ROLES:
        print(f"  {role} dpm form            {d['role_dpm'][role]:+.1f}")

    print("\nPrediction:")
    print(f"  P({blue['team']} wins, blue side) = {res['p_blue']:.1%}")
    print(f"  P({red['team']} wins, red side)   = {res['p_red']:.1%}")
    print(f"  -> {res['winner']} favored ({res['confidence']:.1%})")
    print("\nNote: blue side carries a ~53% base-rate edge; swapping sides shifts the number.")


if __name__ == "__main__":
    main()
