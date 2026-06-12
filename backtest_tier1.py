#!/usr/bin/env python3
"""
backtest_tier1.py - measure the impact of the data-policy fix on international play.

The problem: train_v1.load_and_clean() keeps only datacompleteness=="complete",
which drops 100% of LPL (China) games. Chinese teams therefore arrive at Worlds /
MSI with no rating history - a likely cause of weak international predictions.

The fix (stopgap): the Elo backbone only needs team + side + result, and the LPL
"partial" rows DO carry those. So we rate on a results-only stream restricted to
Tier-1 leagues + international events, LPL included.

This script runs online (leakage-free) Elo WITH vs WITHOUT the LPL data and
compares performance on the SAME international games, plus rating coverage and
the share of cross-region games the form model literally cannot touch.

Run:  python backtest_tier1.py
"""
from __future__ import annotations
import warnings

import numpy as np
import pandas as pd

from backtest import LEAGUE_REGION, MAJOR_REGIONS, _point_metrics
from elo import EloModel
from train_v1 import DATA_DIR

warnings.filterwarnings("ignore")

YEARS = [2023, 2024, 2025, 2026]
INTL = {"WLDs", "MSI", "EWC", "FST"}                       # international events
TIER1 = {"LCK", "LPL", "LEC", "LCS", "LLA", "CBLOL",       # pre-2025 majors
         "PCS", "VCS", "LJL",
         "LTA", "LTA N", "LTA S", "LCP"}                   # 2025+ restructure
IN_SCOPE = TIER1 | INTL
K, HOME_ADV = 40.0, 20.0      # fixed (tuned earlier) so with/without-LPL is comparable
MIN_PRIOR = 10                # a team counts as "rated" after this many games

READ_COLS = ["gameid", "datacompleteness", "league", "date",
             "side", "position", "teamname", "result", "gamelength"]


def load_results(complete_only: bool) -> pd.DataFrame:
    """Results-only team rows in scope. complete_only=True reproduces the old
    data policy (drops LPL); False keeps partial rows (recovers LPL)."""
    frames = [pd.read_csv(DATA_DIR / f"{y}_LoL_esports_match_data_from_OraclesElixir.csv",
                          usecols=READ_COLS, low_memory=False) for y in YEARS]
    df = pd.concat(frames, ignore_index=True)
    df = df[df["position"] == "team"]
    df = df[df["result"].isin([0, 1])]
    df = df[(df["gamelength"].isna()) | (df["gamelength"] > 900)]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "gameid", "teamname", "side"])
    df = df[df["league"].isin(IN_SCOPE)]
    if complete_only:
        df = df[df["datacompleteness"] == "complete"]
    return df


def home_leagues(df: pd.DataFrame) -> pd.Series:
    """Each team's home region = mode of its *domestic* (Tier-1) league."""
    dom = df[df["league"].isin(TIER1)]
    return dom.groupby("teamname")["league"].agg(lambda s: s.mode().iat[0])


def build_stream(df: pd.DataFrame, home: pd.Series) -> pd.DataFrame:
    good = df.groupby("gameid")["side"].transform("nunique").eq(2)
    df = df[good]
    blue = df[df["side"] == "Blue"].rename(columns={"teamname": "blue_team", "result": "blue_result"})
    red = df[df["side"] == "Red"].rename(columns={"teamname": "red_team"})
    g = (blue[["gameid", "date", "league", "blue_team", "blue_result"]]
         .merge(red[["gameid", "red_team"]], on="gameid", how="inner"))
    g["target"] = g["blue_result"].astype(int)
    g["blue_region"] = g["blue_team"].map(home).map(lambda x: LEAGUE_REGION.get(x, "OTHER"))
    g["red_region"] = g["red_team"].map(home).map(lambda x: LEAGUE_REGION.get(x, "OTHER"))
    g["cross_region"] = (
        (g["blue_region"] != g["red_region"])
        & g["blue_region"].isin(MAJOR_REGIONS)
        & g["red_region"].isin(MAJOR_REGIONS)
    )
    g["is_intl"] = g["league"].isin(INTL)
    return g.sort_values(["date", "gameid"]).reset_index(drop=True)


def run_elo(stream: pd.DataFrame) -> pd.DataFrame:
    """Online Elo: predict each game from pre-game ratings, then update.
    Also records how many prior games each team had (rating coverage)."""
    m = EloModel(k=K, home_adv=HOME_ADV)
    seen: dict[str, int] = {}
    probs, bprior, rprior = [], [], []
    for b, r, y in zip(stream["blue_team"], stream["red_team"], stream["target"]):
        bprior.append(seen.get(b, 0))
        rprior.append(seen.get(r, 0))
        probs.append(m.update(b, r, float(y)))
        seen[b] = seen.get(b, 0) + 1
        seen[r] = seen.get(r, 0) + 1
    out = stream.assign(elo_p=probs, blue_prior=bprior, red_prior=rprior)
    out.attrs["model"] = m
    return out


def _row(label, df):
    if len(df) == 0:
        print(f"  {label:<24} (no games)")
        return
    m = _point_metrics(df["target"], df["elo_p"])
    cov = float(((df["blue_prior"] >= MIN_PRIOR) & (df["red_prior"] >= MIN_PRIOR)).mean())
    print(f"  {label:<24} n={len(df):>4}  acc={m['acc']:.3f}  "
          f"log_loss={m['log_loss']:.3f}  brier={m['brier']:.3f}  both_rated={cov:.2f}")


def main() -> None:
    comp = load_results(complete_only=True)    # without LPL (old policy)
    full = load_results(complete_only=False)   # with LPL (results-only)
    home = home_leagues(full)                  # LPL teams get a real home region

    s_comp = run_elo(build_stream(comp, home))
    s_full = run_elo(build_stream(full, home))
    lpl_teams = set(full.loc[full["league"] == "LPL", "teamname"])

    print("[scope] Tier-1 leagues + international events, 2023-2026")
    print(f"  games  WITHOUT LPL (complete-only): {len(s_comp):,}")
    print(f"  games  WITH    LPL (results-only) : {len(s_full):,}  "
          f"(+{len(s_full) - len(s_comp):,}; LPL rows = {int((s_full['league'] == 'LPL').sum()):,})")

    # Compare on the SAME international games (intersection), so only ratings differ
    ic = s_comp[s_comp["is_intl"]].set_index("gameid")
    iff = s_full[s_full["is_intl"]].set_index("gameid")
    common = ic.index.intersection(iff.index)
    extra = len(iff) - len(common)

    print(f"\n[international games]  shared={len(common)}  "
          f"only-in-results-only={extra} (e.g. MSI'24, recovered by the looser filter)")

    print("\n=== ALL INTL (shared games) ===")
    _row("WITHOUT LPL", ic.loc[common])
    _row("WITH LPL", iff.loc[common])

    crc = ic.loc[common][ic.loc[common]["cross_region"]]
    crf = iff.loc[common][iff.loc[common]["cross_region"]]
    print("\n=== CROSS-REGION INTL (shared games) ===")
    _row("WITHOUT LPL", crc)
    _row("WITH LPL", crf)

    # How much of the headline cut can the form model even see?
    cr_full = s_full[(s_full["is_intl"]) & (s_full["cross_region"])]
    if len(cr_full):
        lpl_share = cr_full.apply(
            lambda x: (x["blue_team"] in lpl_teams) or (x["red_team"] in lpl_teams), axis=1).mean()
        print(f"\n[form-model blind spot] cross-region intl games involving an LPL team: "
              f"{lpl_share:.0%} (the v1 feature model cannot score these at all)")

    # Sanity: leaderboard should now be real top teams (no tier-2 inflation)
    model = s_full.attrs["model"]
    counts = pd.concat([s_full["blue_team"], s_full["red_team"]]).value_counts()
    ranked = sorted(((t, r) for t, r in model.ratings.items() if counts.get(t, 0) >= 30),
                    key=lambda x: -x[1])[:12]
    print("\n[elo] top 12 (Tier-1 + intl scope, with LPL, >=30 games):")
    for t, r in ranked:
        tag = "  [LPL]" if t in lpl_teams else ""
        print(f"  {r:7.1f}  {t}{tag}")


if __name__ == "__main__":
    main()
