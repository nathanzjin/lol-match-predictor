#!/usr/bin/env python3
"""
backtest_players.py - does going to the INDIVIDUAL-PLAYER level help?

The project already rates teams (Elo, region-anchored Elo) and tracks team form.
This asks the next question: a team is five players, and rosters move - so do
player-level signals predict better than team-level ones, and do they add
anything on top of a team rating?

Models scored on the SAME time-ordered holdout (80/20 temporal split), all
leakage-free (every rating/stat uses only prior games):

  baselines (single pre-game probability, no training)
    * always-blue   - constant train blue-win base rate
    * team-elo      - plain team Elo (elo.py)
    * region-elo    - region-anchored Elo (region_elo.py); prior headline model
    * player-elo    - Elo keyed on PLAYER, travels through transfers/subs

  trained (XGBoost on a temporal split)
    * player-stats  - per-role rolling individual stats + roster continuity
    * player-full   - player-stats + player-elo diff
    * combined      - player-full + region-elo strength diff
                      (does the player view add over the best team rating?)

Headline buckets mirror backtest.py: overall, intra-region, and international /
cross-region (where the LPL-inclusive player stats should pay off). Plus a
roster-continuity stratification: the core claim is that player-level helps most
exactly when a lineup just changed, so we compare player-elo vs team-elo split
by how many starters carried over from the team's previous game.

Run:  python backtest_players.py
"""
from __future__ import annotations
import warnings

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from player_features import (
    ROLES, PLAYER_STATS, load_player_rows, build_lineups, add_roster_continuity,
    add_player_rolling, build_role_stat_lineups, PlayerElo,
)
from backtest import (_point_metrics, _per_game, _paired_ci, _mcnemar_p,
                      LEAGUE_REGION, MAJOR_REGIONS)
from backtest_tier1 import TIER1, INTL
from elo import EloModel, tune
from region_elo import RegionAnchoredElo

warnings.filterwarnings("ignore")

TRAIN_FRAC = 0.80
SEED = 7
# Player-Elo and region-anchored-Elo params reused by train_v3/predict_v3 so
# the saved model and the live predictor build identical rating features.
PELO_K, PELO_HOME = 24.0, 20.0
RELO_BETA, RELO_KREGION = 1.25, 32.0
XGB_KW = dict(n_estimators=400, learning_rate=0.05, max_depth=4, subsample=0.8,
              colsample_bytree=0.8, min_child_weight=3, eval_metric="logloss",
              n_jobs=-1, random_state=42)


def _xgb() -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", XGBClassifier(**XGB_KW)),
    ])


# --------------------------------------------------------------------------- #
# Assemble one master per-(gameid, team) table, then pair into blue vs red
# --------------------------------------------------------------------------- #
def home_region_map(players: pd.DataFrame) -> dict[str, str]:
    """Each team's home region = region of its most-common domestic (Tier-1) league."""
    dom = players[players["league"].isin(TIER1)].drop_duplicates(["gameid", "teamname"])
    if not len(dom):
        return {}
    home = dom.groupby("teamname")["league"].agg(lambda s: s.mode().iat[0])
    return {t: LEAGUE_REGION.get(lg, "OTHER") for t, lg in home.items()}


def pair_blue_red(team_tbl: pd.DataFrame, value_cols: list[str]) -> pd.DataFrame:
    """Pivot a per-team table to one row per game with blue_/red_ columns."""
    good = team_tbl.groupby("gameid")["side"].transform("nunique").eq(2)
    t = team_tbl[good]
    keep = ["gameid", "date", "league", "result"] + value_cols
    blue = (t[t["side"] == "Blue"][keep]
            .rename(columns={"result": "blue_result",
                             **{c: f"blue_{c}" for c in value_cols}}))
    red = (t[t["side"] == "Red"][["gameid"] + value_cols]
           .rename(columns={c: f"red_{c}" for c in value_cols}))
    g = blue.merge(red, on="gameid", how="inner")
    g["target"] = g["blue_result"].astype(int)
    return g.sort_values(["date", "gameid"]).reset_index(drop=True)


def build_master() -> tuple[pd.DataFrame, list[str]]:
    players = load_player_rows()
    region_of = home_region_map(players)

    # per (gameid, team): roster + meta + continuity
    lineups = add_roster_continuity(build_lineups(players))

    # per (gameid, team): each role-player's leakage-safe rolling stats
    role_tbl = build_role_stat_lineups(add_player_rolling(players))
    roll_cols = [f"{r}_roll_{s}" for r in ROLES for s in PLAYER_STATS]
    career_cols = [f"{r}_career_games" for r in ROLES]

    # merge roster/continuity with role stats on the team-game key
    keep_role = ["gameid", "teamname"] + roll_cols + career_cols
    team_tbl = lineups.merge(role_tbl[keep_role], on=["gameid", "teamname"], how="left")
    team_tbl["region"] = team_tbl["teamname"].map(lambda t: region_of.get(t, "OTHER"))

    value_cols = ["teamname", "region", "roster_continuity"] + ROLES + roll_cols + career_cols
    g = pair_blue_red(team_tbl, value_cols)

    # ---- per-role stat differentials (blue role player - red role player) ---
    feat = []
    for r in ROLES:
        for s in PLAYER_STATS:
            col = f"diff_{r}_{s}"
            g[col] = g[f"blue_{r}_roll_{s}"] - g[f"red_{r}_roll_{s}"]
            feat.append(col)
        cc = f"diff_{r}_career"
        g[cc] = g[f"blue_{r}_career_games"] - g[f"red_{r}_career_games"]
        feat.append(cc)
    # roster continuity: each side + the lower of the two (a changed lineup on
    # either side is the interesting case)
    g["blue_continuity"] = g["blue_roster_continuity"]
    g["red_continuity"] = g["red_roster_continuity"]
    g["min_continuity"] = g[["blue_continuity", "red_continuity"]].min(axis=1)
    feat += ["blue_continuity", "red_continuity", "min_continuity"]

    # region / international labels for bucketing
    g["cross_region"] = ((g["blue_region"] != g["red_region"])
                         & g["blue_region"].isin(MAJOR_REGIONS)
                         & g["red_region"].isin(MAJOR_REGIONS))
    g["is_intl"] = g["league"].isin(INTL)
    return g, feat


# --------------------------------------------------------------------------- #
# Rating replays (online, leakage-free) over the paired game table
# --------------------------------------------------------------------------- #
def add_player_elo(g: pd.DataFrame, **kw) -> pd.DataFrame:
    m = PlayerElo(**kw)
    bs, rs, p = [], [], []
    for row in g.itertuples(index=False):
        blue = [getattr(row, f"blue_{r}") for r in ROLES]
        red = [getattr(row, f"red_{r}") for r in ROLES]
        bs.append(m.team_strength(blue)); rs.append(m.team_strength(red))
        p.append(m.update(blue, red, float(row.target)))
    g = g.assign(blue_pelo=bs, red_pelo=rs,
                 pelo_diff=np.array(bs) - np.array(rs), pelo_p=p)
    g.attrs["pelo"] = m
    return g


def add_team_elo(g: pd.DataFrame, k: float, home_adv: float) -> pd.DataFrame:
    m = EloModel(k=k, home_adv=home_adv)
    diff, p = [], []
    for row in g.itertuples(index=False):
        b, r = row.blue_teamname, row.red_teamname
        diff.append((m.rating(b) + m.home_adv) - m.rating(r))
        p.append(m.update(b, r, float(row.target)))
    return g.assign(telo_diff=diff, telo_p=p)


def add_region_elo(g: pd.DataFrame, beta: float, k_region: float) -> pd.DataFrame:
    m = RegionAnchoredElo(beta=beta, k_region=k_region)
    diff, p = [], []
    for row in g.itertuples(index=False):
        b, br, r, rr = row.blue_teamname, row.blue_region, row.red_teamname, row.red_region
        diff.append(m.effective(b, br) - m.effective(r, rr))
        p.append(m.update(b, br, r, rr, float(row.target)))
    g = g.assign(relo_diff=diff, relo_p=p)
    g.attrs["relo"] = m
    return g


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def report(title: str, y, preds: dict[str, np.ndarray], compare: tuple[str, str] | None):
    n = len(y)
    print(f"\n=== {title}  (n={n}) ===")
    if n == 0:
        print("  (no games)"); return
    for name, p in preds.items():
        m = _point_metrics(y, p)
        auc = " n/a" if np.isnan(m["auc"]) else f"{m['auc']:.3f}"
        print(f"  {name:<13} acc={m['acc']:.3f}  log_loss={m['log_loss']:.3f}  auc={auc}  brier={m['brier']:.3f}")
    if compare and n >= 30:
        a, b = compare
        ca, la = _per_game(y, preds[a])
        cb, lb = _per_game(y, preds[b])
        dacc, alo, ahi = _paired_ci(ca, cb, SEED)
        dll, llo, lhi = _paired_ci(la, lb, SEED + 1)
        pm = _mcnemar_p(ca, cb)
        sig = "  <- significant" if not (alo <= 0 <= ahi) else "  (CI incl 0)"
        print(f"  [{a} - {b}]  d_acc {dacc:+.3f} [{alo:+.3f},{ahi:+.3f}] (McNemar p={pm:.3f}){sig}")
        print(f"  {' ' * len(f'[{a} - {b}]')}  d_log_loss {dll:+.3f} [{llo:+.3f},{lhi:+.3f}]")


def main() -> None:
    g, feat = build_master()
    print(f"[games] paired games with player features: {len(g):,}")
    print(f"[features] {len(feat)} player features "
          f"({len(ROLES)} roles x {len(PLAYER_STATS)} stats + career + continuity)")

    split = int(len(g) * TRAIN_FRAC)
    test_start = g.iloc[split]["date"]

    # --- tune team-Elo on pre-test games only (same discipline as backtest.py)
    pre = g[g["date"] < test_start]
    best = tune(pre["blue_teamname"], pre["red_teamname"], pre["target"])
    print(f"[team-elo] tuned K={best['k']:.0f}, home_adv={best['home_adv']:.0f}")

    # --- rating replays over the full ordered stream
    g = add_team_elo(g, k=best["k"], home_adv=best["home_adv"])
    g = add_player_elo(g, k=PELO_K, home_adv=PELO_HOME)
    g = add_region_elo(g, beta=RELO_BETA, k_region=RELO_KREGION)

    train, test = g.iloc[:split], g.iloc[split:]
    print(f"[split] train {len(train):,} ({train['date'].min().date()} -> {train['date'].max().date()})"
          f" | test {len(test):,} ({test['date'].min().date()} -> {test['date'].max().date()})")
    base_rate = float(train["target"].mean())

    # --- trained models
    stats_pipe = _xgb().fit(train[feat], train["target"])
    full_cols = feat + ["pelo_diff"]
    full_pipe = _xgb().fit(train[full_cols], train["target"])
    comb_cols = full_cols + ["relo_diff"]
    comb_pipe = _xgb().fit(train[comb_cols], train["target"])

    y = test["target"].to_numpy()
    preds = {
        "always-blue": np.full(len(test), base_rate),
        "team-elo": test["telo_p"].to_numpy(),
        "region-elo": test["relo_p"].to_numpy(),
        "player-elo": test["pelo_p"].to_numpy(),
        "player-stats": stats_pipe.predict_proba(test[feat])[:, 1],
        "player-full": full_pipe.predict_proba(test[full_cols])[:, 1],
        "combined": comb_pipe.predict_proba(test[comb_cols])[:, 1],
    }

    def subset(mask):
        m = mask.to_numpy()
        return y[m], {k: v[m] for k, v in preds.items()}

    yo, po = y, preds
    report("OVERALL", yo, po, ("player-elo", "team-elo"))
    report("OVERALL (trained models)", yo, po, ("combined", "player-full"))

    ys, ps = subset(~test["cross_region"])
    report("INTRA-REGION", ys, ps, ("player-elo", "team-elo"))

    yi, pi = subset(test["is_intl"])
    report("INTERNATIONAL EVENTS", yi, pi, ("player-elo", "team-elo"))

    yc, pc = subset(test["cross_region"])
    report("CROSS-REGION", yc, pc, ("combined", "region-elo"))

    # --- the core claim: player-level helps most when the lineup just changed
    print("\n\n########## ROSTER-CHANGE STRATIFICATION ##########")
    print("player-elo vs team-elo, split by min roster continuity (starters kept "
          "from each team's previous game). Low continuity = a lineup the team-\n"
          "level rating hasn't seen play together yet.")
    for lo, hi, label in [(0, 3, "<=3 of 5 kept (changed lineup)"),
                          (4, 4, "4 of 5 kept (one swap)"),
                          (5, 5, "5 of 5 kept (stable lineup)")]:
        mask = (test["min_continuity"] >= lo) & (test["min_continuity"] <= hi)
        ys, ps = subset(mask)
        report(f"CONTINUITY {label}", ys,
               {"team-elo": ps["team-elo"], "player-elo": ps["player-elo"]},
               ("player-elo", "team-elo"))

    # --- feature importance for the combined model
    imp = comb_pipe.named_steps["model"].feature_importances_
    print("\n[combined] top 15 features by importance:")
    for f, v in sorted(zip(comb_cols, imp), key=lambda x: -x[1])[:15]:
        tag = ""
        if f == "relo_diff": tag = "  <- region rating"
        elif f == "pelo_diff": tag = "  <- player Elo"
        print(f"  {f:<22} {v:.4f}{tag}")


if __name__ == "__main__":
    main()
