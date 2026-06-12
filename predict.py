#!/usr/bin/env python3
"""
predict.py - run the trained v1 model on a hypothetical blue-vs-red matchup.

It rebuilds the SAME no-leakage rolling features that train_v1.py uses: for each
team we take the mean of their most recent N games (default 10), then feed the
blue-minus-red differentials (+ patch) to the saved pipeline.

Examples:
  python predict.py "T1" "Gen.G"
  python predict.py "T1" "Gen.G" --patch 16.12
  python predict.py --list                 # print every known team name
  python predict.py --list | grep -i fnatic
"""
from __future__ import annotations
import argparse
import difflib
import sys

import joblib
import pandas as pd

# Reuse training constants + the player->team aggregation so features can't drift.
from train_v1 import (
    DATA_DIR, MODEL_DIR, USE_COLS, ROLL_MAP,
    ROLLING_WINDOW, MIN_GAMES, build_team_frame,
)

# Include 2026 here (train_v1 holds it out of training) so "current form" is current.
ALL_YEARS = [2023, 2024, 2025, 2026]


def load_clean(years: list[int]) -> pd.DataFrame:
    """Same load + cleaning as train_v1.load_and_clean, but over chosen years."""
    frames = []
    for y in years:
        path = DATA_DIR / f"{y}_LoL_esports_match_data_from_OraclesElixir.csv"
        if path.exists():
            frames.append(pd.read_csv(path, usecols=USE_COLS, low_memory=False))
    if not frames:
        sys.exit(f"No data CSVs found in {DATA_DIR}/. Run:  python download_data.py")
    df = pd.concat(frames, ignore_index=True)
    df = df[df["datacompleteness"] == "complete"]
    df = df[df["result"].isin([0, 1])]
    df = df[df["gamelength"] > 900]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["patch"] = pd.to_numeric(df["patch"], errors="coerce")
    df = df.dropna(subset=["date", "gameid", "teamname", "side"])
    return df


def resolve_team(name: str, available: list[str]) -> str:
    """Exact -> case-insensitive -> fuzzy match, with a helpful error otherwise."""
    if name in available:
        return name
    lower = {t.lower(): t for t in available}
    if name.lower() in lower:
        return lower[name.lower()]
    close = difflib.get_close_matches(name, available, n=5, cutoff=0.6)
    msg = f"Team '{name}' not found."
    if close:
        msg += " Did you mean: " + ", ".join(repr(c) for c in close) + "?"
    msg += "\nUse --list to see all team names."
    sys.exit(msg)


def team_form(teams: pd.DataFrame, name: str, window: int, min_games: int):
    """Mean of a team's last `window` games for each rolling source column."""
    sub = teams[teams["teamname"] == name].sort_values(["date", "gameid"]).tail(window)
    n = len(sub)
    if n < min_games:
        sys.exit(f"Not enough history for '{name}': {n} game(s), need >= {min_games}.")
    form = {out: float(sub[src].mean()) for src, out in ROLL_MAP.items()}
    return form, n, sub["date"].max().date()


def main() -> None:
    ap = argparse.ArgumentParser(description="Predict a LoL match outcome (blue vs red).")
    ap.add_argument("blue", nargs="?", help="Blue-side team name")
    ap.add_argument("red", nargs="?", help="Red-side team name")
    ap.add_argument("--patch", type=float, default=None,
                    help="Patch number (default: latest patch present in the data)")
    ap.add_argument("--window", type=int, default=ROLLING_WINDOW,
                    help=f"Games of history per team (default {ROLLING_WINDOW})")
    ap.add_argument("--min-games", type=int, default=MIN_GAMES,
                    help=f"Minimum games of history required (default {MIN_GAMES})")
    ap.add_argument("--years", type=int, nargs="+", default=ALL_YEARS,
                    help="Years of data to load for form (default 2023 2024 2025 2026)")
    ap.add_argument("--list", action="store_true", help="List known team names and exit")
    args = ap.parse_args()

    model_path = MODEL_DIR / "lol_pipeline_v1.joblib"
    feat_path = MODEL_DIR / "feature_cols.joblib"
    if not model_path.exists() or not feat_path.exists():
        sys.exit("Model artifacts missing. Train first:  python train_v1.py")

    df = load_clean(args.years)
    teams = build_team_frame(df)
    available = sorted(teams["teamname"].unique())

    if args.list:
        print("\n".join(available))
        return

    if not args.blue or not args.red:
        sys.exit('Provide two teams:  python predict.py "<blue>" "<red>"   (or --list)')

    blue = resolve_team(args.blue, available)
    red = resolve_team(args.red, available)
    if blue == red:
        sys.exit("Blue and red must be different teams.")

    pipe = joblib.load(model_path)
    features = joblib.load(feat_path)

    blue_form, blue_n, blue_last = team_form(teams, blue, args.window, args.min_games)
    red_form, red_n, red_last = team_form(teams, red, args.window, args.min_games)

    patch = args.patch if args.patch is not None else float(df["patch"].dropna().max())

    row = {f"diff_{out}": blue_form[out] - red_form[out] for out in ROLL_MAP.values()}
    row["patch"] = patch
    X = pd.DataFrame([row])[features]  # enforce exact training column order

    p_blue = float(pipe.predict_proba(X)[:, 1][0])
    p_red = 1.0 - p_blue
    winner = blue if p_blue >= 0.5 else red
    conf = max(p_blue, p_red)

    print(f"\nMatchup  (patch {patch:g})")
    print(f"  BLUE  {blue}   - form from last {blue_n} games (thru {blue_last})")
    print(f"  RED   {red}   - form from last {red_n} games (thru {red_last})")

    print("\nFeature differentials (blue - red):")
    for out in ROLL_MAP.values():
        print(f"  diff_{out:<16} {row['diff_' + out]:+.3f}")

    print("\nPrediction:")
    print(f"  P({blue} wins, blue side) = {p_blue:.1%}")
    print(f"  P({red} wins, red side)   = {p_red:.1%}")
    print(f"  -> {winner} favored ({conf:.1%})")
    print("\nNote: blue side carries a ~53% base-rate edge, so swapping sides changes the number.")


if __name__ == "__main__":
    main()
