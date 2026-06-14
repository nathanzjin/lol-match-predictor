#!/usr/bin/env python3
"""
season.py - the model's track record over the current season, graded honestly.

The arbitrary-matchup predictor (predictor.py) uses the final trained model.
This module answers a different question: "how is the model actually doing on
this season's games as results come in?" - and it must be leakage-free to mean
anything.

How it stays honest:
  * Ratings are replayed online over the whole history (predict each game from
    pre-game ratings, then update), so every feature for a 2026 game uses only
    earlier games.
  * The XGBoost is trained only on games BEFORE the season starts, then asked to
    predict each season game it has never seen.

So this is a true walk-forward: for every Tier-1 matchup of the season the model
commits to a probability, and we grade it against the real result. As fresh data
is downloaded through the year, re-running extends the track record.

Output (JSON-ready) feeds the /api/performance endpoint and the dashboard:
  * summary       - n, accuracy, log-loss, Brier, blue-side baseline
  * timeline      - cumulative accuracy + log-loss after each game (the chart)
  * by_league     - per-league accuracy
  * games         - the graded game log (most recent first)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from backtest import _point_metrics
from backtest_players import (
    build_master, add_player_elo, add_player_elo_region, add_region_elo, _xgb,
    PELO_K, PELO_HOME, RELO_BETA, RELO_KREGION, PELO_RA_BETA, PELO_RA_KREGION,
)
from player_features import MAJOR_REGIONS


def _summary(df: pd.DataFrame) -> dict:
    if not len(df):
        return {"n": 0}
    y = df["target"].to_numpy()
    p = df["p_blue"].to_numpy()
    m = _point_metrics(y, p)
    base = max(y.mean(), 1 - y.mean())   # always-pick-the-more-common-side
    return {
        "n": int(len(df)),
        "correct": int((df["pred_blue"] == (y == 1)).sum()),
        "accuracy": round(float(m["acc"]), 4),
        "log_loss": round(float(m["log_loss"]), 4),
        "brier": round(float(m["brier"]), 4),
        "auc": None if np.isnan(m["auc"]) else round(float(m["auc"]), 4),
        "baseline_acc": round(float(base), 4),
    }


def compute_season_performance(season_year: int | None = None,
                               recent_games: int = 150) -> dict:
    """Walk-forward track record for the latest (or given) season's Tier-1 games."""
    g, feat, _ = build_master()
    g = add_player_elo(g, k=PELO_K, home_adv=PELO_HOME)
    g = add_player_elo_region(g, beta=PELO_RA_BETA, k_region=PELO_RA_KREGION)
    g = add_region_elo(g, beta=RELO_BETA, k_region=RELO_KREGION)
    cols = feat + ["pelo_diff", "pelo_ra_diff", "relo_diff"]

    g["year"] = g["date"].dt.year
    if season_year is None:
        season_year = int(g["year"].max())
    season_start = pd.Timestamp(f"{season_year}-01-01")

    train = g[g["date"] < season_start]
    pipe = _xgb().fit(train[cols], train["target"])

    # Tier-1 matchups of the season (the supported scope) in date order
    season = g[(g["year"] == season_year)
               & g["blue_region"].isin(MAJOR_REGIONS)
               & g["red_region"].isin(MAJOR_REGIONS)].copy()
    season = season.sort_values(["date", "gameid"]).reset_index(drop=True)
    if not len(season):
        return {"season_year": season_year, "summary": {"all": {"n": 0}},
                "timeline": [], "by_league": [], "games": [],
                "trained_through": str(train["date"].max().date()) if len(train) else None}

    season["p_blue"] = pipe.predict_proba(season[cols])[:, 1]
    season["pred_blue"] = season["p_blue"] >= 0.5
    season["actual_blue"] = season["target"] == 1
    season["correct"] = season["pred_blue"] == season["actual_blue"]

    # cumulative timeline (chart): accuracy + log-loss after each game
    y = season["target"].to_numpy()
    p = np.clip(season["p_blue"].to_numpy(), 1e-15, 1 - 1e-15)
    correct_cum = np.cumsum(season["correct"].to_numpy()) / np.arange(1, len(season) + 1)
    ll_each = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    ll_cum = np.cumsum(ll_each) / np.arange(1, len(season) + 1)
    timeline = [
        {"i": int(i + 1), "date": d.strftime("%Y-%m-%d"),
         "cum_acc": round(float(correct_cum[i]), 4),
         "cum_log_loss": round(float(ll_cum[i]), 4)}
        for i, d in enumerate(season["date"])
    ]

    by_league = []
    for lg, sub in season.groupby("league"):
        s = _summary(sub)
        s["league"] = lg
        by_league.append(s)
    by_league.sort(key=lambda x: -x["n"])

    games = []
    for r in season.sort_values(["date", "gameid"], ascending=False).head(recent_games).itertuples(index=False):
        blue_won = bool(r.target == 1)
        games.append({
            "date": r.date.strftime("%Y-%m-%d"),
            "league": r.league,
            "blue": r.blue_teamname, "red": r.red_teamname,
            "blue_region": r.blue_region, "red_region": r.red_region,
            "p_blue": round(float(r.p_blue), 4),
            "predicted": r.blue_teamname if r.p_blue >= 0.5 else r.red_teamname,
            "actual": r.blue_teamname if blue_won else r.red_teamname,
            "correct": bool(r.correct),
        })

    return {
        "season_year": season_year,
        "trained_through": str(train["date"].max().date()),
        "data_through": str(season["date"].max().date()),
        "summary": {"all": _summary(season)},
        "timeline": timeline,
        "by_league": by_league,
        "games": games,
    }


if __name__ == "__main__":
    import json
    perf = compute_season_performance()
    s = perf["summary"]["all"]
    print(f"Season {perf['season_year']} (trained through {perf['trained_through']}, "
          f"data through {perf['data_through']})")
    print(f"  Tier-1 games graded: {s['n']}")
    print(f"  accuracy={s['accuracy']:.3f}  log_loss={s['log_loss']:.3f}  "
          f"brier={s['brier']:.3f}  baseline={s['baseline_acc']:.3f}")
    print("\n  by league:")
    for b in perf["by_league"]:
        print(f"    {b['league']:<8} n={b['n']:>4}  acc={b['accuracy']:.3f}")
    print("\n  most recent 5 graded games:")
    for gme in perf["games"][:5]:
        mark = "OK " if gme["correct"] else "XX "
        print(f"    {mark} {gme['date']}  {gme['blue']} vs {gme['red']}  "
              f"P(blue)={gme['p_blue']:.0%}  -> pred {gme['predicted']}, actual {gme['actual']}")
