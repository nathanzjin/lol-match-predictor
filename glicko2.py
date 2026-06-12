#!/usr/bin/env python3
"""
glicko2.py - region-anchored Glicko-2 (uncertainty-aware ratings).

region_elo.py fixed the cross-region *scale* problem (accuracy gain proven) but
its *calibration* gain (log-loss) stayed inside the noise. Glicko-2 adds a
rating deviation (RD) per team: teams rarely tested - new rosters, wildcard
regions, anyone idle through an off-season - keep a high RD, so their predicted
win probabilities are pulled toward 50% via Glicko's g() factor instead of
being confidently wrong. That is exactly the cross-region calibration lever.

Design mirrors region_elo.py so the comparison is clean:
  * team skill  -> Glicko-2 (mu, phi=RD, sigma=volatility), updated only on
                   intra-region games; idle RD inflation per week.
  * region term -> simple online rating, updated only on cross-region games.
  * effective   -> team + beta*(region - base); prediction shrinks toward 0.5
                   using BOTH teams' RD.

Online-per-game Glicko-2 variant with weekly idle inflation, so it slots into
the leakage-free predict-then-update stream. Evaluated with the same
expanding-window walk-forward as backtest_rolling.py, head-to-head vs
region-anchored Elo.

Run:  python glicko2.py
"""
from __future__ import annotations
import math
import warnings

import numpy as np
import pandas as pd

from backtest import _point_metrics, _per_game, _paired_ci, _mcnemar_p, MAJOR_REGIONS
from backtest_tier1 import load_results, home_leagues, build_stream, run_elo
from region_elo import run_region, BETAS, K_REGIONS

warnings.filterwarnings("ignore")

SCALE = 173.7178
BASE = 1500.0
DEFAULT_RD = 350.0
DEFAULT_VOL = 0.06
TAU = 0.5
MAX_PHI = DEFAULT_RD / SCALE
PERIOD_DAYS = 7

FOLD_YEARS = [2023, 2024, 2025, 2026]
MIN_TUNE = 40
G_BETAS = (0.5, 1.0, 1.5)      # smaller grid (Glicko update is heavier)
G_KREG = (24.0, 32.0)


def _g(phi: float) -> float:
    return 1.0 / math.sqrt(1.0 + 3.0 * phi * phi / (math.pi * math.pi))


class GlickoRegion:
    def __init__(self, beta=1.0, k_region=24.0, home_adv=20.0, tau=TAU):
        self.beta = beta
        self.k_region = k_region
        self.home_mu = home_adv / SCALE
        self.tau = tau
        self.team: dict[str, list] = {}      # team -> [mu, phi, sigma, last_period]
        self.region: dict[str, float] = {}   # region -> rating (points)

    def _t(self, team: str, period: int) -> list:
        if team not in self.team:
            self.team[team] = [0.0, DEFAULT_RD / SCALE, DEFAULT_VOL, period]
        return self.team[team]

    def _reg(self, reg: str) -> float:
        return self.region.get(reg, BASE)

    def _inflate(self, st: list, period: int) -> None:
        idle = period - st[3]
        if idle > 0:
            st[1] = min(math.sqrt(st[1] ** 2 + idle * st[2] ** 2), MAX_PHI)
            st[3] = period

    def predict(self, b, breg, r, rreg, period) -> float:
        sb, sr = self._t(b, period), self._t(r, period)
        self._inflate(sb, period)
        self._inflate(sr, period)
        mb = sb[0] + self.beta * (self._reg(breg) - BASE) / SCALE + self.home_mu
        mr = sr[0] + self.beta * (self._reg(rreg) - BASE) / SCALE
        g = _g(math.sqrt(sb[1] ** 2 + sr[1] ** 2))
        return 1.0 / (1.0 + math.exp(-g * (mb - mr)))

    def _update_one(self, st: list, mu_opp: float, phi_opp: float, score: float) -> None:
        mu, phi, sigma, _ = st
        g = _g(phi_opp)
        E = 1.0 / (1.0 + math.exp(-g * (mu - mu_opp)))
        E = min(max(E, 1e-6), 1 - 1e-6)
        v = 1.0 / (g * g * E * (1 - E))
        delta = v * g * (score - E)
        a = math.log(sigma * sigma)

        def f(x: float) -> float:
            ex = math.exp(x)
            return (ex * (delta * delta - phi * phi - v - ex)
                    / (2.0 * (phi * phi + v + ex) ** 2)) - (x - a) / (self.tau * self.tau)

        A = a
        if delta * delta > phi * phi + v:
            B = math.log(delta * delta - phi * phi - v)
        else:
            k = 1
            while f(a - k * self.tau) < 0:
                k += 1
            B = a - k * self.tau
        fA, fB = f(A), f(B)
        for _ in range(100):
            if abs(B - A) <= 1e-6:
                break
            C = A + (A - B) * fA / (fB - fA)
            fC = f(C)
            if fC * fB <= 0:
                A, fA = B, fB
            else:
                fA = fA / 2.0
            B, fB = C, fC
        sigma_new = math.exp(A / 2.0)
        phi_star = math.sqrt(phi * phi + sigma_new * sigma_new)
        phi_new = 1.0 / math.sqrt(1.0 / (phi_star * phi_star) + 1.0 / v)
        st[0] = mu + phi_new * phi_new * g * (score - E)
        st[1] = phi_new
        st[2] = sigma_new

    def observe(self, b, breg, r, rreg, blue_won, period, p_pred) -> None:
        sb, sr = self._t(b, period), self._t(r, period)
        is_cross = (breg != rreg) and (breg in MAJOR_REGIONS) and (rreg in MAJOR_REGIONS)
        if is_cross:                                  # region rating moves; team skill held
            d = self.k_region * (blue_won - p_pred)
            self.region[breg] = self._reg(breg) + d
            self.region[rreg] = self._reg(rreg) - d
            sb[3] = sr[3] = period
        else:                                         # within-region Glicko-2 update
            mb, pb, mr, pr = sb[0], sb[1], sr[0], sr[1]
            self._update_one(sb, mr, pr, blue_won)
            self._update_one(sr, mb, pb, 1.0 - blue_won)


def run_glicko(stream: pd.DataFrame, beta: float, k_region: float,
               home_adv: float = 20.0, tau: float = TAU) -> pd.DataFrame:
    m = GlickoRegion(beta=beta, k_region=k_region, home_adv=home_adv, tau=tau)
    epoch = stream["date"].min()
    preds = []
    for date, b, br, r, rr, y in zip(stream["date"], stream["blue_team"], stream["blue_region"],
                                     stream["red_team"], stream["red_region"], stream["target"]):
        period = (date - epoch).days // PERIOD_DAYS
        p = m.predict(b, br, r, rr, period)
        preds.append(p)
        m.observe(b, br, r, rr, float(y), period, p)
    out = stream.assign(gl_p=preds)
    out.attrs["model"] = m
    return out


def _ll(y, p) -> float:
    y = np.asarray(y, float)
    p = np.clip(np.asarray(p, float), 1e-15, 1 - 1e-15)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _tune(run_fn, before, betas, kregs):
    best = None
    for beta in betas:
        for kr in kregs:
            run = run_fn(before, beta, kr)
            sel = run[run["cross_region"]]
            col = "gl_p" if "gl_p" in sel else "reg_p"
            ll = _ll(sel["target"], sel[col])
            if best is None or ll < best[0]:
                best = (ll, beta, kr)
    return best[1], best[2]


def main() -> None:
    df = load_results(complete_only=False)
    stream = build_stream(df, home_leagues(df))
    plain = run_elo(stream).set_index("gameid")["elo_p"]

    rows = []
    print("[folds] expanding-window walk-forward (region-Elo vs region-Glicko2)")
    for Y in FOLD_YEARS:
        start, end = pd.Timestamp(f"{Y}-01-01"), pd.Timestamp(f"{Y + 1}-01-01")
        before = stream[stream["date"] < start]
        enough = len(before[before["cross_region"]]) >= MIN_TUNE

        if enough:
            eb, ek = _tune(run_region, before, BETAS, K_REGIONS)
            gb, gk = _tune(run_glicko, before, G_BETAS, G_KREG)
        else:
            eb, ek = 1.0, 24.0
            gb, gk = 1.0, 24.0

        reg_p = run_region(stream, eb, ek).set_index("gameid")["reg_p"]
        gl_p = run_glicko(stream, gb, gk).set_index("gameid")["gl_p"]
        mask = (stream["date"] >= start) & (stream["date"] < end) & stream["cross_region"]
        ids = stream.loc[mask, "gameid"]
        if len(ids):
            rows.append(pd.DataFrame({
                "fold": Y,
                "y": stream.loc[mask, "target"].to_numpy(),
                "plain": plain.loc[ids].to_numpy(),
                "region": reg_p.loc[ids].to_numpy(),
                "glicko": gl_p.loc[ids].to_numpy(),
            }))
        print(f"  {Y}: n={len(ids):>3} | elo(beta={eb},kr={ek})  glicko(beta={gb},kr={gk})"
              f"{'' if enough else '  [defaults]'}")

    pool = pd.concat(rows, ignore_index=True)
    y = pool["y"].to_numpy()
    print(f"\n=== POOLED CROSS-REGION INTL (n={len(pool)}) ===")
    for name, col in [("plain Elo", "plain"), ("region Elo", "region"), ("region Glicko-2", "glicko")]:
        m = _point_metrics(y, pool[col].to_numpy())
        conf = float(np.mean(np.abs(pool[col].to_numpy() - 0.5)) + 0.5)  # mean confidence (sharpness)
        print(f"  {name:<16} acc={m['acc']:.3f}  log_loss={m['log_loss']:.3f}  "
              f"auc={m['auc']:.3f}  brier={m['brier']:.3f}  mean_conf={conf:.3f}")

    # The key question: does Glicko-2 uncertainty beat region Elo on calibration?
    cr_g, lr_g = _per_game(y, pool["glicko"].to_numpy())
    cr_e, lr_e = _per_game(y, pool["region"].to_numpy())
    dacc, alo, ahi = _paired_ci(cr_g, cr_e, 21)       # glicko - region
    dll, llo, lhi = _paired_ci(lr_g, lr_e, 22)
    pm = _mcnemar_p(cr_g, cr_e)
    sa = "  <- significant" if not (alo <= 0 <= ahi) else "  (CI includes 0)"
    sl = "  <- significant" if not (llo <= 0 <= lhi) else "  (CI includes 0)"
    print(f"\n  Glicko-2 - region Elo:  d_acc {dacc:+.3f} [{alo:+.3f},{ahi:+.3f}] "
          f"(McNemar p={pm:.3f}){sa}")
    print(f"                          d_log_loss {dll:+.3f} [{llo:+.3f},{lhi:+.3f}]{sl}")

    print("\n[per-fold log_loss]")
    for Y, grp in pool.groupby("fold"):
        le = _point_metrics(grp["y"], grp["region"])["log_loss"]
        lg = _point_metrics(grp["y"], grp["glicko"])["log_loss"]
        print(f"  {Y}:  region={le:.3f}  glicko={lg:.3f}  (n={len(grp)})")


if __name__ == "__main__":
    main()
