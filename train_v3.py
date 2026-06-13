#!/usr/bin/env python3
"""
train_v3.py - fit and persist the player-level model (the new project best).

backtest_players.py showed that individual-player signals beat the team-level
ratings: player-Elo (keyed on the player, so it follows transfers/subs) plus
per-role rolling box-score stats, with the region-anchored team rating folded
in, reached the best holdout numbers the project has produced.

This trains that "combined" model and saves everything predict_v3.py needs:
  * the fitted XGBoost pipeline + its feature order
  * the final player-Elo ratings (per player)
  * the final region-anchored-Elo state (team + region tables)
  * the params used to build them (so the live predictor matches training)

It first reports a temporal-holdout score (train on the first 80% of games,
score the rest) for an honest, leakage-free read on quality, then refits on ALL
games so the saved ratings/model are as current as the data allows.

Run:  python train_v3.py
"""
from __future__ import annotations
import warnings

import joblib
import numpy as np

from train_v1 import MODEL_DIR
from backtest import _point_metrics
from backtest_players import (
    build_master, add_player_elo, add_region_elo, _xgb,
    TRAIN_FRAC, PELO_K, PELO_HOME, RELO_BETA, RELO_KREGION,
)
from player_features import ROLES, PLAYER_STATS

warnings.filterwarnings("ignore")


def main() -> None:
    MODEL_DIR.mkdir(exist_ok=True)

    g, feat = build_master()
    # rating features (online => leakage-free); keep the fitted models for saving
    g = add_player_elo(g, k=PELO_K, home_adv=PELO_HOME)
    g = add_region_elo(g, beta=RELO_BETA, k_region=RELO_KREGION)
    pelo = g.attrs["pelo"]
    relo = g.attrs["relo"]

    cols = feat + ["pelo_diff", "relo_diff"]
    print(f"[data] games={len(g):,}  features={len(cols)}")

    # 1) Honest temporal-holdout score
    split = int(len(g) * TRAIN_FRAC)
    train, test = g.iloc[:split], g.iloc[split:]
    pipe = _xgb().fit(train[cols], train["target"])
    proba = pipe.predict_proba(test[cols])[:, 1]
    m = _point_metrics(test["target"].to_numpy(), proba)
    print(f"[holdout] train {len(train):,} ({train['date'].min().date()}->{train['date'].max().date()})"
          f" | test {len(test):,} ({test['date'].min().date()}->{test['date'].max().date()})")
    print(f"[holdout] acc={m['acc']:.4f}  log_loss={m['log_loss']:.4f}  "
          f"auc={m['auc']:.4f}  brier={m['brier']:.4f}")

    # 2) Refit on ALL games for the production artifact
    final = _xgb().fit(g[cols], g["target"])

    joblib.dump(final, MODEL_DIR / "lol_pipeline_v3.joblib")
    joblib.dump(cols, MODEL_DIR / "feature_cols_v3.joblib")
    joblib.dump({
        "player_ratings": dict(pelo.ratings),
        "pelo_params": dict(base=pelo.base, k=pelo.k, home_adv=pelo.home_adv, scale=pelo.scale),
        "region_team": dict(relo.team),
        "region_region": dict(relo.region),
        "relo_params": dict(base=relo.base, scale=relo.scale, beta=relo.beta,
                            k_region=relo.k_region, k_team=relo.k_team, home_adv=relo.home_adv),
        "roles": ROLES,
        "player_stats": PLAYER_STATS,
    }, MODEL_DIR / "rating_state_v3.joblib")

    print(f"[save] {MODEL_DIR / 'lol_pipeline_v3.joblib'}")
    print(f"[save] {MODEL_DIR / 'feature_cols_v3.joblib'}")
    print(f"[save] {MODEL_DIR / 'rating_state_v3.joblib'}  "
          f"({len(pelo.ratings):,} player ratings, {len(relo.region)} regions)")

    # Flavor: top players by final Elo (>= a sensible game count would be nicer,
    # but the ratings dict alone is enough to eyeball the leaderboard)
    top = sorted(pelo.ratings.items(), key=lambda x: -x[1])[:15]
    print("\n[player-elo] top 15 by final rating:")
    for name, r in top:
        print(f"  {r:7.1f}  {name}")


if __name__ == "__main__":
    main()
