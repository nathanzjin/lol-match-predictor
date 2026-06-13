#!/usr/bin/env python3
"""
backtest_rolling.py - expanding-window walk-forward over years.

region_elo.py evaluated only on 2025-2026 (tune <2025, test >=2025), leaving
just ~230 cross-region international games - too few to prove the region-anchor
gain is real. This pools cross-region games from EVERY year by walking forward:

  for each fold year Y in 2023..2026:
      tune (beta, k_region) on cross-region games strictly before Y
        (fallback to a fixed prior when there's not enough history yet),
      run region-anchored + plain Elo ONLINE over the full stream with those
        params, and collect predictions for Y's cross-region intl games.

Because Elo predicts before it updates, every per-game prediction is leakage-
free; only the hyperparameters look at pre-fold outcomes. Pooling the folds
roughly triples the cross-region sample, so the paired test (region vs plain)
has a real chance to clear significance.

Run:  python backtest_rolling.py
"""
from __future__ import annotations
import warnings

import numpy as np
import pandas as pd

from backtest import _point_metrics, _per_game, _paired_ci, _mcnemar_p
from backtest_tier1 import load_results, home_leagues, build_stream, run_elo
from region_elo import run_region, BETAS, K_REGIONS

warnings.filterwarnings("ignore")

FOLD_YEARS = [2023, 2024, 2025, 2026]
MIN_TUNE = 40                       # cross-region games needed before we tune
DEFAULT_BETA, DEFAULT_KR = 1.0, 24.0


def _ll(y, p) -> float:
    y = np.asarray(y, float)
    p = np.clip(np.asarray(p, float), 1e-15, 1 - 1e-15)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def tune_region(before: pd.DataFrame):
    """Pick beta/k_region minimising online cross-region log-loss on pre-fold data."""
    best = None
    for beta in BETAS:
        for kr in K_REGIONS:
            run = run_region(before, beta, kr)
            sel = run[run["cross_region"]]
            ll = _ll(sel["target"], sel["reg_p"])
            if best is None or ll < best[0]:
                best = (ll, beta, kr)
    return best[1], best[2]


def main() -> None:
    df = load_results(complete_only=False)
    stream = build_stream(df, home_leagues(df))
    plain = run_elo(stream).set_index("gameid")["elo_p"]   # team-only baseline (fixed params)

    collected = []
    print("[folds] expanding-window walk-forward (cross-region intl games per fold)")
    for Y in FOLD_YEARS:
        start = pd.Timestamp(f"{Y}-01-01")
        end = pd.Timestamp(f"{Y + 1}-01-01")
        before = stream[stream["date"] < start]
        xr_before = before[before["cross_region"]]

        if len(xr_before) >= MIN_TUNE:
            beta, kr = tune_region(before)
            tuned = f"tuned beta={beta}, k_region={kr} (on {len(xr_before)} prior xr games)"
        else:
            beta, kr = DEFAULT_BETA, DEFAULT_KR
            tuned = f"default beta={beta}, k_region={kr} (only {len(xr_before)} prior xr games)"

        reg_p = run_region(stream, beta, kr).set_index("gameid")["reg_p"]
        fold_mask = (stream["date"] >= start) & (stream["date"] < end) & stream["cross_region"]
        ids = stream.loc[fold_mask, "gameid"]
        if len(ids):
            collected.append(pd.DataFrame({
                "gameid": ids.to_numpy(),
                "fold": Y,
                "y": stream.loc[fold_mask, "target"].to_numpy(),
                "plain": plain.loc[ids].to_numpy(),
                "region": reg_p.loc[ids].to_numpy(),
            }))
        print(f"  {Y}:  n={len(ids):>3}  | {tuned}")

    pool = pd.concat(collected, ignore_index=True)
    y = pool["y"].to_numpy()
    mp = _point_metrics(y, pool["plain"].to_numpy())
    mr = _point_metrics(y, pool["region"].to_numpy())

    print(f"\n=== POOLED CROSS-REGION INTL (all folds, n={len(pool)}) ===")
    print(f"  plain Elo        acc={mp['acc']:.3f}  log_loss={mp['log_loss']:.3f}  "
          f"auc={mp['auc']:.3f}  brier={mp['brier']:.3f}")
    print(f"  region-anchored  acc={mr['acc']:.3f}  log_loss={mr['log_loss']:.3f}  "
          f"auc={mr['auc']:.3f}  brier={mr['brier']:.3f}")

    cp, lp = _per_game(y, pool["plain"].to_numpy())
    cr, lr = _per_game(y, pool["region"].to_numpy())
    dacc, alo, ahi = _paired_ci(cr, cp, 11)          # region - plain (+ve favours region)
    dll, llo, lhi = _paired_ci(lr, lp, 12)           # region - plain (-ve favours region)
    pm = _mcnemar_p(cr, cp)
    sig_a = "  <- significant" if not (alo <= 0 <= ahi) else "  (CI includes 0)"
    sig_l = "  <- significant" if not (llo <= 0 <= lhi) else "  (CI includes 0)"
    print(f"\n  region - plain:  d_acc {dacc:+.3f} [{alo:+.3f},{ahi:+.3f}] "
          f"(McNemar p={pm:.3f}){sig_a}")
    print(f"                   d_log_loss {dll:+.3f} [{llo:+.3f},{lhi:+.3f}]{sig_l}")

    print("\n[per-fold accuracy]")
    for Y, grp in pool.groupby("fold"):
        a_p = _point_metrics(grp["y"], grp["plain"])["acc"]
        a_r = _point_metrics(grp["y"], grp["region"])["acc"]
        print(f"  {Y}:  plain={a_p:.3f}  region={a_r:.3f}  (n={len(grp)})")


if __name__ == "__main__":
    main()
