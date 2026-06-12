#!/usr/bin/env python3
"""
backtest.py - walk-forward evaluation harness for LoL match prediction.

Scores three models on the SAME held-out, time-ordered test set:
  * always-blue  - constant baseline at the training blue-win base rate
  * elo          - pre-game Elo ratings (elo.py), tuned on training games only
  * v1           - the saved XGBoost rolling-form pipeline (models/)

Robust cross-league analysis:
  - Each game is labelled with a *time-aware* home league per team (mode of that
    team's trailing games, so promotions/relocations and the event itself don't
    pollute the label) and bucketed:
      intra_league | intl_event | regional_mixed | cross_league_other
  - A curated league->region map adds a cross_region flag; the headline metric
    is international-event games between teams from different major regions.
  - Every bucket reports bootstrap 95% CIs on accuracy and log-loss, plus a
    paired elo-vs-v1 comparison (difference CI + McNemar test) so we don't
    over-read small gaps on small samples.

Methodology / leakage discipline:
  - Games ordered by date, split 80/20 in time (same split as train_v1).
  - Elo predicts each game strictly before updating; K/home-adv tuned only on
    pre-test games. Evaluation set is the v1-usable games (post cold-start).

Run:  python backtest.py
"""
from __future__ import annotations
import warnings
from collections import Counter, deque

import joblib
import numpy as np
import pandas as pd
from scipy.stats import rankdata, binomtest

from train_v1 import (
    load_and_clean, build_team_frame, add_rolling, build_games,
    MODEL_DIR, TRAIN_FRAC,
)
from elo import EloModel, tune

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
HOME_WINDOW = 20        # trailing games used to infer a team's home league
N_BOOT = 2000           # bootstrap resamples
SEED = 42
MIN_N_CI = 30           # below this, skip CIs / paired tests for a bucket

# Curated league -> region map. Deliberately limited to leagues whose teams
# actually attend international events (where cross_region matters and is
# unambiguous). Everything else is "OTHER" and never counts as cross_region.
# Easy to extend; the harness prints any unmapped leagues seen in the test set.
LEAGUE_REGION = {
    # Korea
    "LCK": "KR", "LCKC": "KR", "KeSPA": "KR",
    # China
    "LPL": "CN", "LDL": "CN",
    # EMEA
    "LEC": "EMEA", "EM": "EMEA", "LFL": "EMEA", "PRM": "EMEA", "NLC": "EMEA",
    "UL": "EMEA", "ESLOL": "EMEA", "LVP SL": "EMEA", "TCL": "EMEA", "AL": "EMEA",
    "HM": "EMEA", "EBL": "EMEA",
    # Americas
    "LCS": "AMERICAS", "LTA": "AMERICAS", "LTA N": "AMERICAS", "LTA S": "AMERICAS",
    "LLA": "AMERICAS", "CBLOL": "AMERICAS", "NACL": "AMERICAS",
    # Asia-Pacific
    "PCS": "APAC", "VCS": "APAC", "LJL": "APAC", "LCP": "APAC", "LCO": "APAC",
    # International events
    "WLDs": "INTL", "MSI": "INTL", "EWC": "INTL",
}
MAJOR_REGIONS = {"KR", "CN", "EMEA", "AMERICAS", "APAC"}
INTL_EVENTS = {"WLDs", "MSI", "EWC"}        # best-vs-best across regions
REGIONAL_MIXED = {"EM", "LTA", "LCP"}       # competitions that span sub-leagues

MODEL_COLS = {"always-blue": "always_blue", "elo": "elo", "v1": "v1"}


# --------------------------------------------------------------------------- #
# Time-aware home league + buckets
# --------------------------------------------------------------------------- #
def _trailing_mode(s: pd.Series, window: int) -> pd.Series:
    """Most common league among the previous `window` games (current excluded).

    Cold-start (no prior games) falls back to the current league. Leakage-safe:
    only games strictly before the current one inform the label.
    """
    out, dq, cnt = [], deque(), Counter()
    for val in s:
        out.append(cnt.most_common(1)[0][0] if cnt else val)
        dq.append(val)
        cnt[val] += 1
        if len(dq) > window:
            old = dq.popleft()
            cnt[old] -= 1
            if cnt[old] == 0:
                del cnt[old]
    return pd.Series(out, index=s.index)


def build_game_stream(teams: pd.DataFrame) -> pd.DataFrame:
    """One row per game (blue vs red), every 2-sided game, sorted by date,
    with time-aware home leagues, regions, a cross_region flag, and a bucket."""
    # Time-aware home league at the team-row level
    tr = teams[["gameid", "teamname", "date", "league"]].copy()
    tr = tr.sort_values(["teamname", "date", "gameid"])
    tr["home_league"] = (tr.groupby("teamname", sort=False)["league"]
                         .transform(lambda s: _trailing_mode(s, HOME_WINDOW)))
    home_tbl = tr[["gameid", "teamname", "home_league"]]

    cols = ["gameid", "date", "side", "teamname", "result", "league"]
    t = teams[cols].copy()
    blue = (t[t["side"] == "Blue"]
            .rename(columns={"teamname": "blue_team", "result": "blue_result"}))
    red = (t[t["side"] == "Red"]
           .rename(columns={"teamname": "red_team", "result": "red_result"}))
    g = (blue[["gameid", "date", "league", "blue_team", "blue_result"]]
         .merge(red[["gameid", "red_team"]], on="gameid", how="inner"))
    g["target"] = g["blue_result"].astype(int)

    # attach home leagues for each side
    g = g.merge(home_tbl.rename(columns={"teamname": "blue_team",
                                         "home_league": "blue_home"}),
                on=["gameid", "blue_team"], how="left")
    g = g.merge(home_tbl.rename(columns={"teamname": "red_team",
                                         "home_league": "red_home"}),
                on=["gameid", "red_team"], how="left")

    g["blue_region"] = g["blue_home"].map(lambda x: LEAGUE_REGION.get(x, "OTHER"))
    g["red_region"] = g["red_home"].map(lambda x: LEAGUE_REGION.get(x, "OTHER"))
    g["cross_region"] = (
        (g["blue_region"] != g["red_region"])
        & g["blue_region"].isin(MAJOR_REGIONS)
        & g["red_region"].isin(MAJOR_REGIONS)
    )

    def bucket(row) -> str:
        if row["league"] in INTL_EVENTS:
            return "intl_event"
        if row["league"] in REGIONAL_MIXED:
            return "regional_mixed"
        if row["blue_home"] == row["red_home"]:
            return "intra_league"
        return "cross_league_other"

    g["bucket"] = g.apply(bucket, axis=1)
    return g.sort_values(["date", "gameid"]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Metrics + statistics
# --------------------------------------------------------------------------- #
def _point_metrics(y, p) -> dict:
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), 1e-15, 1 - 1e-15)
    pred = (p >= 0.5).astype(int)
    acc = float(np.mean(pred == y))
    ll = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    brier = float(np.mean((p - y) ** 2))
    if 0 < y.sum() < len(y):
        ranks = rankdata(p)  # average ranks -> ties handled, constant = 0.5
        n_pos = y.sum()
        n_neg = len(y) - n_pos
        auc = float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))
    else:
        auc = float("nan")
    return {"acc": acc, "log_loss": ll, "auc": auc, "brier": brier}


def _per_game(y, p):
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), 1e-15, 1 - 1e-15)
    correct = ((p >= 0.5).astype(int) == y.astype(int)).astype(float)
    loss = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    return correct, loss


def _boot_ci(arr, seed):
    n = len(arr)
    if n < MIN_N_CI:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    means = arr[rng.integers(0, n, size=(N_BOOT, n))].mean(axis=1)
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def _paired_ci(a, b, seed):
    """Bootstrap CI of mean(a - b), paired over the same games."""
    d = a - b
    point = float(d.mean())
    n = len(d)
    if n < MIN_N_CI:
        return (point, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, n, size=(N_BOOT, n))].mean(axis=1)
    return (point, float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def _mcnemar_p(correct_a, correct_b) -> float:
    b = int(np.sum((correct_a == 1) & (correct_b == 0)))
    c = int(np.sum((correct_a == 0) & (correct_b == 1)))
    if b + c == 0:
        return 1.0
    return float(binomtest(min(b, c), b + c, 0.5).pvalue)


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def report_bucket(title: str, df: pd.DataFrame) -> None:
    n = len(df)
    print(f"\n{title}  (n={n})")
    if n == 0:
        print("  (no games)")
        return
    small = n < MIN_N_CI
    if small:
        print("  (too few games for reliable CIs; point estimates only)")
    print(f"  {'model':<12} {'acc [95% CI]':<24} {'log_loss [95% CI]':<24} {'auc':>6} {'brier':>7}")

    y = df["y"].to_numpy()
    pg = {}
    for name, col in MODEL_COLS.items():
        p = df[col].to_numpy()
        m = _point_metrics(y, p)
        correct, loss = _per_game(y, p)
        pg[name] = (correct, loss)
        if small:
            acc_s, ll_s = f"{m['acc']:.3f}", f"{m['log_loss']:.3f}"
        else:
            alo, ahi = _boot_ci(correct, SEED)
            llo, lhi = _boot_ci(loss, SEED + 1)
            acc_s = f"{m['acc']:.3f} [{alo:.3f},{ahi:.3f}]"
            ll_s = f"{m['log_loss']:.3f} [{llo:.3f},{lhi:.3f}]"
        auc_s = " n/a" if np.isnan(m["auc"]) else f"{m['auc']:.3f}"
        print(f"  {name:<12} {acc_s:<24} {ll_s:<24} {auc_s:>6} {m['brier']:>7.3f}")

    if not small:
        ce, le = pg["elo"]
        cv, lv = pg["v1"]
        dacc, alo, ahi = _paired_ci(ce, cv, SEED + 2)        # +ve favours elo
        dll, llo, lhi = _paired_ci(le, lv, SEED + 3)         # -ve favours elo
        pm = _mcnemar_p(ce, cv)
        sig_a = "" if (alo <= 0 <= ahi) else "  <- significant"
        sig_l = "" if (llo <= 0 <= lhi) else "  <- significant"
        print(f"  elo vs v1:  d_acc {dacc:+.3f} [{alo:+.3f},{ahi:+.3f}] "
              f"(McNemar p={pm:.3f}){sig_a}")
        print(f"              d_log_loss {dll:+.3f} [{llo:+.3f},{lhi:+.3f}]{sig_l}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    teams = build_team_frame(load_and_clean())
    stream = build_game_stream(teams)
    games, _ = build_games(add_rolling(teams))  # cold-start filtered, date-sorted

    feat_path = MODEL_DIR / "feature_cols.joblib"
    model_path = MODEL_DIR / "lol_pipeline_v1.joblib"
    if not model_path.exists() or not feat_path.exists():
        raise SystemExit("Missing model artifacts. Train first:  python train_v1.py")
    features = joblib.load(feat_path)
    pipe = joblib.load(model_path)

    split = int(len(games) * TRAIN_FRAC)
    train, test = games.iloc[:split], games.iloc[split:]
    test_start = test["date"].min()
    print(f"[data] usable games: {len(games):,} | stream games: {len(stream):,}")
    print(f"[split] train {len(train):,} ({train['date'].min().date()} -> {train['date'].max().date()})"
          f" | test {len(test):,} ({test['date'].min().date()} -> {test['date'].max().date()})")

    # Elo: tune on pre-test stream, run the full stream
    train_stream = stream[stream["date"] < test_start]
    best = tune(train_stream["blue_team"], train_stream["red_team"], train_stream["target"])
    print(f"[elo] tuned on {len(train_stream):,} pre-test games: "
          f"K={best['k']:.0f}, home_adv={best['home_adv']:.0f} "
          f"(train log_loss {best['train_log_loss']:.4f})")
    elo = EloModel(k=best["k"], home_adv=best["home_adv"])
    stream = stream.assign(elo_p=elo.fit_predict(
        stream["blue_team"], stream["red_team"], stream["target"]))

    # Shared test predictions
    by_id = stream.set_index("gameid")
    base_rate = float(train["target"].mean())
    preds = pd.DataFrame({
        "y": test["target"].to_numpy(),
        "bucket": test["gameid"].map(by_id["bucket"]).to_numpy(),
        "cross_region": test["gameid"].map(by_id["cross_region"]).fillna(False).to_numpy(),
        "always_blue": np.full(len(test), base_rate),
        "elo": test["gameid"].map(by_id["elo_p"]).to_numpy(),
        "v1": pipe.predict_proba(test[features])[:, 1],
    })

    print(f"\n[base rate] train blue-win = {base_rate:.4f} | test blue-win = {test['target'].mean():.4f}")
    print("\n[bucket sizes in test]")
    for b, c in preds["bucket"].value_counts().items():
        print(f"  {b:<20} {c:>5}")

    report_bucket("=== OVERALL (test set) ===", preds)
    for b in ["intra_league", "intl_event", "regional_mixed", "cross_league_other"]:
        report_bucket(f"=== {b.upper()} ===", preds[preds["bucket"] == b])

    # Headline: true cross-region matchups at international events
    headline = preds[(preds["bucket"] == "intl_event") & (preds["cross_region"])]
    report_bucket("=== HEADLINE: INTL EVENT, CROSS-REGION ===", headline)

    # Diagnostics: which leagues in the test set are unmapped (region OTHER)?
    test_ids = set(test["gameid"])
    sub = stream[stream["gameid"].isin(test_ids)]
    unmapped = sorted(set(sub.loc[sub["blue_region"] == "OTHER", "blue_home"]).union(
        sub.loc[sub["red_region"] == "OTHER", "red_home"]).difference({np.nan}))
    if unmapped:
        print(f"\n[diagnostic] {len(unmapped)} unmapped league(s) in test "
              f"(region=OTHER, never counted as cross_region):")
        print("  " + ", ".join(map(str, unmapped)))

    # Flavor: Elo leaderboard among well-connected teams
    counts = pd.concat([stream["blue_team"], stream["red_team"]]).value_counts()
    ranked = sorted(((tm, r) for tm, r in elo.ratings.items() if counts.get(tm, 0) >= 50),
                    key=lambda x: -x[1])[:10]
    print("\n[elo] top 10 teams by final rating (>=50 games):")
    for tm, r in ranked:
        print(f"  {r:7.1f}  {tm}")


if __name__ == "__main__":
    main()
