#!/usr/bin/env python3
"""
train_v3.py - fit and persist the player-level model (the new project best).

backtest_players.py showed that individual-player signals beat the team-level
ratings, and that region-anchoring the player Elo (splitting within-region skill
from region strength, like region_elo.py one level down) fixes its weak-region
inflation and rescues cross-region prediction. The best model ("combined-both")
feeds XGBoost BOTH player ratings - plain player-Elo for the overall edge and
the region-anchored one for cross-region robustness - plus per-role rolling
box-score stats and the region-anchored team rating.

This trains that model and saves everything predict_v3.py needs:
  * the fitted XGBoost pipeline + its feature order
  * the final plain player-Elo ratings (per player)
  * the final region-anchored player-Elo state (player + region tables)
  * the final region-anchored-Elo team state (team + region tables)
  * each player's modal region + the params used (so the live predictor matches)

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
    build_master, add_player_elo, add_player_elo_region, add_region_elo,
    tune_player_elo_region, _xgb, TRAIN_FRAC, PELO_K, PELO_HOME,
    RELO_BETA, RELO_KREGION, PELO_RA_BETA, PELO_RA_KREGION,
)
from player_features import ROLES, PLAYER_STATS

warnings.filterwarnings("ignore")


def main() -> None:
    MODEL_DIR.mkdir(exist_ok=True)

    g, feat, info = build_master()

    # region-anchored player Elo: tune beta/k_region on ALL cross-region games
    # (production rating wants the best region calibration the data allows)
    bra = tune_player_elo_region(g, np.ones(len(g), dtype=bool))
    if bra:
        _, ra_beta, ra_kregion = bra
        print(f"[player-elo-ra] tuned beta={ra_beta}, k_region={ra_kregion:.0f} "
              f"(cross-region log_loss {bra[0]:.4f})")
    else:
        ra_beta, ra_kregion = PELO_RA_BETA, PELO_RA_KREGION

    # rating features (online => leakage-free); keep the fitted models for saving
    g = add_player_elo(g, k=PELO_K, home_adv=PELO_HOME)
    g = add_player_elo_region(g, beta=ra_beta, k_region=ra_kregion)
    g = add_region_elo(g, beta=RELO_BETA, k_region=RELO_KREGION)
    pelo = g.attrs["pelo"]
    pelo_ra = g.attrs["pelo_ra"]
    relo = g.attrs["relo"]

    # the project-best "combined-both" feature set: plain player-Elo (overall
    # edge) + region-anchored player-Elo (cross-region robustness) + region rating
    cols = feat + ["pelo_diff", "pelo_ra_diff", "relo_diff"]
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
        "player_ra_player": dict(pelo_ra.player),
        "player_ra_region": dict(pelo_ra.region),
        "pelo_ra_params": dict(base=pelo_ra.base, k=pelo_ra.k, k_region=pelo_ra.k_region,
                              beta=pelo_ra.beta, home_adv=pelo_ra.home_adv, scale=pelo_ra.scale),
        "region_team": dict(relo.team),
        "region_region": dict(relo.region),
        "relo_params": dict(base=relo.base, scale=relo.scale, beta=relo.beta,
                            k_region=relo.k_region, k_team=relo.k_team, home_adv=relo.home_adv),
        "player_region": info["player_region"],
        "roles": ROLES,
        "player_stats": PLAYER_STATS,
    }, MODEL_DIR / "rating_state_v3.joblib")

    print(f"[save] {MODEL_DIR / 'lol_pipeline_v3.joblib'}")
    print(f"[save] {MODEL_DIR / 'feature_cols_v3.joblib'}")
    print(f"[save] {MODEL_DIR / 'rating_state_v3.joblib'}  "
          f"({len(pelo.ratings):,} player ratings, {len(relo.region)} regions)")

    # Flavor: top players by region-anchored EFFECTIVE rating (the fixed
    # leaderboard - real stars rather than weak-region farmers)
    pr, pg = info["player_region"], info["player_games"]
    elig = [p for p in pelo_ra.player if pg.get(p, 0) >= 100]
    eff = lambda p: pelo_ra.player_effective(p, pr.get(p, "OTHER"))
    print("\n[region-anchored effective] top 15 (>=100 games):")
    for p in sorted(elig, key=lambda x: -eff(x))[:15]:
        print(f"  {eff(p):7.1f}  {p:<16} ({pr.get(p, '?')}, {pg[p]}g)")


if __name__ == "__main__":
    main()
