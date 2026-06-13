#!/usr/bin/env python3
"""
region_elo.py - region-anchored Elo (GPR-style team + region blend).

Plain Elo can't compare weakly-connected leagues: a team that farms a weak
region inflates its rating with no strong opponents to lose points to, so
cross-region predictions are unreliable (we measured ~base-rate log-loss).

This model splits strength into two levels, like Riot's Global Power Rankings
(a weighted blend of a team's Elo and its league/region Elo):

    effective(team) = team_elo[team] + beta * (region_elo[region] - BASE)

  * team_elo   - within-region skill, updated ONLY on intra-region games, so
                 each region's team ratings stay zero-sum around BASE (a team's
                 deviation from its own region's average).
  * region_elo - relative region strength, updated ONLY on cross-region
                 (international) games - the only games that compare regions.

The region term is identical for two same-region teams, so it cancels on
intra-region games (prediction == plain team Elo there) and only bites on
cross-region matchups - exactly where plain Elo failed. A cold-start team
inherits its region's strength as a prior, which helps newly-seen
international teams (e.g. a fresh LPL roster arriving at Worlds).

Leakage-free: predicts from pre-game ratings, then updates. Tuned on pre-2025
games and evaluated on 2025-2026, so the comparison is out-of-sample.

Run:  python region_elo.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from backtest import _point_metrics, _per_game, _paired_ci, _mcnemar_p, MAJOR_REGIONS
from backtest_tier1 import load_results, home_leagues, build_stream, run_elo

BASE, SCALE = 1500.0, 400.0
K_TEAM, HOME_ADV = 40.0, 20.0
TRAIN_END = pd.Timestamp("2025-01-01")     # tune on <2025, evaluate on >=2025
BETAS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5)
K_REGIONS = (8, 16, 24, 32, 48)


class RegionAnchoredElo:
    def __init__(self, beta=1.0, k_region=24.0, k_team=K_TEAM,
                 home_adv=HOME_ADV, base=BASE, scale=SCALE, major_regions=None):
        self.beta = beta
        self.k_region = k_region
        self.k_team = k_team
        self.home_adv = home_adv
        self.base = base
        self.scale = scale
        # Which region labels count as "major" for cross-region updates. Defaults
        # to this module's MAJOR_REGIONS; callers using a different region
        # taxonomy (e.g. the Tier-1 player model) pass their own set.
        self.major_regions = MAJOR_REGIONS if major_regions is None else major_regions
        self.team: dict[str, float] = {}
        self.region: dict[str, float] = {}

    def _t(self, x: str) -> float:
        return self.team.get(x, self.base)

    def _r(self, reg: str) -> float:
        return self.region.get(reg, self.base)

    def effective(self, team: str, reg: str) -> float:
        return self._t(team) + self.beta * (self._r(reg) - self.base)

    def expect_blue(self, b, breg, r, rreg) -> float:
        diff = (self.effective(b, breg) + self.home_adv) - self.effective(r, rreg)
        return 1.0 / (1.0 + 10.0 ** (-diff / self.scale))

    def update(self, b, breg, r, rreg, blue_won) -> float:
        p = self.expect_blue(b, breg, r, rreg)
        is_cross = (breg != rreg) and (breg in self.major_regions) and (rreg in self.major_regions)
        if is_cross:
            d = self.k_region * (blue_won - p)            # only region ratings move
            self.region[breg] = self._r(breg) + d
            self.region[rreg] = self._r(rreg) - d
        else:
            d = self.k_team * (blue_won - p)              # only team ratings move
            self.team[b] = self._t(b) + d
            self.team[r] = self._t(r) - d
        return p


def run_region(stream: pd.DataFrame, beta: float, k_region: float) -> pd.DataFrame:
    m = RegionAnchoredElo(beta=beta, k_region=k_region)
    preds = [m.update(b, br, r, rr, float(y)) for b, br, r, rr, y in zip(
        stream["blue_team"], stream["blue_region"],
        stream["red_team"], stream["red_region"], stream["target"])]
    out = stream.assign(reg_p=preds)
    out.attrs["model"] = m
    return out


def _ll(y, p) -> float:
    y = np.asarray(y, float)
    p = np.clip(np.asarray(p, float), 1e-15, 1 - 1e-15)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def main() -> None:
    stream = build_stream(load_results(complete_only=False),
                          home_leagues(load_results(complete_only=False)))

    plain_p = run_elo(stream).set_index("gameid")["elo_p"]   # team-only baseline

    is_xr = stream["cross_region"]
    is_intl = stream["is_intl"]
    train_mask = stream["date"] < TRAIN_END

    # Tune beta / k_region on pre-2025 cross-region games (online preds are
    # leakage-free; only the choice of hyperparams sees these outcomes).
    best = None
    for beta in BETAS:
        for kr in K_REGIONS:
            run = run_region(stream, beta, kr)
            sel = run[train_mask & is_xr]
            ll = _ll(sel["target"], sel["reg_p"])
            if best is None or ll < best[0]:
                best = (ll, beta, kr, run)
    _, beta, kr, best_run = best
    print(f"[tune] best on pre-2025 cross-region games: beta={beta}, k_region={kr} "
          f"(train xr log_loss {best[0]:.4f})")
    reg_p = best_run.set_index("gameid")["reg_p"]

    test = stream[~train_mask]

    def evalset(mask: pd.Series, title: str) -> None:
        sub = test[mask.loc[test.index]]
        ids = sub["gameid"]
        y = sub["target"].to_numpy()
        pp = plain_p.loc[ids].to_numpy()
        rp = reg_p.loc[ids].to_numpy()
        mp, mr = _point_metrics(y, pp), _point_metrics(y, rp)
        print(f"\n=== {title}  (test era >=2025, n={len(sub)}) ===")
        print(f"  plain Elo        acc={mp['acc']:.3f}  log_loss={mp['log_loss']:.3f}  brier={mp['brier']:.3f}")
        print(f"  region-anchored  acc={mr['acc']:.3f}  log_loss={mr['log_loss']:.3f}  brier={mr['brier']:.3f}")
        if len(sub) >= 30:
            cp, lp = _per_game(y, pp)
            cr, lr = _per_game(y, rp)
            dacc, alo, ahi = _paired_ci(cr, cp, 7)        # region - plain (+ve favours region)
            dll, llo, lhi = _paired_ci(lr, lp, 8)         # region - plain (-ve favours region)
            pm = _mcnemar_p(cr, cp)
            sig = "  <- significant" if not (alo <= 0 <= ahi) else ""
            print(f"  region-plain:  d_acc {dacc:+.3f} [{alo:+.3f},{ahi:+.3f}] (McNemar p={pm:.3f}){sig}")
            print(f"                 d_log_loss {dll:+.3f} [{llo:+.3f},{lhi:+.3f}]")

    evalset(is_intl & is_xr, "CROSS-REGION INTL")
    evalset(is_intl, "ALL INTL")
    evalset(~stream["cross_region"], "INTRA-REGION (sanity: should not hurt)")

    m = best_run.attrs["model"]
    print("\n[region ratings] relative strength (base 1500):")
    for reg, r in sorted(m.region.items(), key=lambda x: -x[1]):
        print(f"  {reg:<10} {r:7.1f}")


if __name__ == "__main__":
    main()
