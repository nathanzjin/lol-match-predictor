#!/usr/bin/env python3
"""
train_v2.py - combined model: v1 rolling-form features + region-anchored rating.

v1 used only recent-form differentials; the rating work showed opponent strength
is the bigger signal. This folds them together: every v1 form feature PLUS a
single `rating_diff` = effective(blue) - effective(red) from the region-anchored
Elo (region_elo.py), where effective = team_rating + beta*(region - base). That
one feature carries both team skill and league/region strength, so no manual
league tiering is needed.

To isolate the effect of adding the rating, all three models are trained/scored
on the SAME Tier-1 + international games with the SAME temporal split:
  * form-only      - v1 feature set, retrained on this scope (apples-to-apples)
  * region rating  - the rating model's own pre-game probability (no form)
  * combined       - form features + rating_diff

Leakage discipline: rating_diff is computed online (pre-game) over the rating
stream; the rolling-form features already exclude the current game.

Run:  python train_v2.py
"""
from __future__ import annotations
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from train_v1 import load_and_clean, build_team_frame, add_rolling, build_games, MODEL_DIR
from backtest import (_point_metrics, _per_game, _paired_ci, _mcnemar_p,
                      LEAGUE_REGION, MAJOR_REGIONS)
from backtest_tier1 import load_results, home_leagues, build_stream, INTL, IN_SCOPE
from region_elo import RegionAnchoredElo

warnings.filterwarnings("ignore")

BETA, K_REGION = 1.25, 32.0       # region-anchor params that tuned well pre-2025
TRAIN_FRAC = 0.80
XGB_KW = dict(n_estimators=400, learning_rate=0.05, max_depth=4, subsample=0.8,
              colsample_bytree=0.8, min_child_weight=3, eval_metric="logloss",
              n_jobs=-1, random_state=42)


def _xgb():
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", XGBClassifier(**XGB_KW)),
    ])


def rating_features(stream: pd.DataFrame, beta: float, k_region: float):
    """Replay region-anchored Elo; record pre-game rating_diff + prob per game."""
    m = RegionAnchoredElo(beta=beta, k_region=k_region)
    diff, prob = {}, {}
    for gid, b, br, r, rr, y in zip(stream["gameid"], stream["blue_team"], stream["blue_region"],
                                    stream["red_team"], stream["red_region"], stream["target"]):
        diff[gid] = m.effective(b, br) - m.effective(r, rr)   # no home term: pure strength
        prob[gid] = m.expect_blue(b, br, r, rr)
        m.update(b, br, r, rr, float(y))
    return pd.Series(diff), pd.Series(prob), m


def main() -> None:
    # 1) Form-feature games (v1 pipeline), restricted to Tier-1 + international
    teams = build_team_frame(load_and_clean())
    games, features = build_games(add_rolling(teams))
    league_of = teams.drop_duplicates("gameid").set_index("gameid")["league"]
    games["league"] = games["gameid"].map(league_of)
    games = games[games["league"].isin(IN_SCOPE)].copy()

    # 2) Rating stream (results-only, LPL included) -> rating_diff + region prob
    res = load_results(complete_only=False)
    home = home_leagues(res)
    stream = build_stream(res, home)
    rdiff, rprob, model = rating_features(stream, BETA, K_REGION)
    games["rating_diff"] = games["gameid"].map(rdiff)
    games["reg_p"] = games["gameid"].map(rprob)
    games = games.dropna(subset=["rating_diff", "reg_p"]).reset_index(drop=True)

    # 3) Region / bucket labels for the form games
    def reg(t):
        return LEAGUE_REGION.get(home.get(t, None), "OTHER")
    games["blue_region"] = games["blue_teamname"].map(reg)
    games["red_region"] = games["red_teamname"].map(reg)
    games["cross_region"] = ((games["blue_region"] != games["red_region"])
                             & games["blue_region"].isin(MAJOR_REGIONS)
                             & games["red_region"].isin(MAJOR_REGIONS))
    games["is_intl"] = games["league"].isin(INTL)

    games = games.sort_values(["date", "gameid"]).reset_index(drop=True)
    split = int(len(games) * TRAIN_FRAC)
    train, test = games.iloc[:split], games.iloc[split:]
    print(f"[scope] Tier-1 + intl form games: {len(games):,}")
    print(f"[split] train {len(train):,} ({train['date'].min().date()} -> {train['date'].max().date()})"
          f" | test {len(test):,} ({test['date'].min().date()} -> {test['date'].max().date()})")

    # 4) Train form-only and combined on identical rows
    combined_cols = features + ["rating_diff"]
    form_pipe = _xgb()
    comb_pipe = _xgb()
    form_pipe.fit(train[features], train["target"])
    comb_pipe.fit(train[combined_cols], train["target"])

    p_form = pd.Series(form_pipe.predict_proba(test[features])[:, 1], index=test.index)
    p_comb = pd.Series(comb_pipe.predict_proba(test[combined_cols])[:, 1], index=test.index)

    def block(title, mask=None):
        sl = test if mask is None else test[mask]
        yy = sl["target"].to_numpy()
        rows = {
            "form-only": p_form.loc[sl.index].to_numpy(),
            "region rating": sl["reg_p"].to_numpy(),
            "combined": p_comb.loc[sl.index].to_numpy(),
        }
        print(f"\n=== {title}  (n={len(sl)}) ===")
        for nm, pp in rows.items():
            m = _point_metrics(yy, pp)
            auc = "  n/a" if np.isnan(m["auc"]) else f"{m['auc']:.3f}"
            print(f"  {nm:<14} acc={m['acc']:.3f}  log_loss={m['log_loss']:.3f}  auc={auc}  brier={m['brier']:.3f}")
        if len(sl) >= 30:
            cc, lc = _per_game(yy, rows["combined"])
            cf, lf = _per_game(yy, rows["form-only"])
            da, alo, ahi = _paired_ci(cc, cf, 31)
            dl, llo, lhi = _paired_ci(lc, lf, 32)
            pm = _mcnemar_p(cc, cf)
            sa = "  <- significant" if not (alo <= 0 <= ahi) else "  (CI incl 0)"
            print(f"  combined - form:  d_acc {da:+.3f} [{alo:+.3f},{ahi:+.3f}] (McNemar p={pm:.3f}){sa}")
            print(f"                    d_log_loss {dl:+.3f} [{llo:+.3f},{lhi:+.3f}]")

    block("OVERALL")
    block("INTRA-REGION", ~test["cross_region"])
    block("INTERNATIONAL EVENTS", test["is_intl"])

    # 5) Feature importance: how much does the model lean on rating_diff?
    imp = comb_pipe.named_steps["model"].feature_importances_
    print("\n[combined] feature importance:")
    for f, v in sorted(zip(combined_cols, imp), key=lambda x: -x[1]):
        star = "  <- rating" if f == "rating_diff" else ""
        print(f"  {f:<22} {v:.4f}{star}")

    # 6) Data-driven league strength (avg team rating), the "classification"
    rows = [(home.get(t), st) for t, st in model.team.items() if home.get(t) in IN_SCOPE]
    ls = pd.DataFrame(rows, columns=["league", "rating"]).groupby("league")["rating"]
    tbl = ls.agg(["mean", "count"]).sort_values("mean", ascending=False)
    print("\n[league strength] mean team rating (Tier-1 + intl leagues, >=3 teams):")
    for lg, row in tbl[tbl["count"] >= 3].head(12).iterrows():
        print(f"  {lg:<8} {row['mean']:7.1f}  ({int(row['count'])} teams)")
    print("\n[region strength] learned region offsets:")
    for rg, rv in sorted(model.region.items(), key=lambda x: -x[1]):
        print(f"  {rg:<10} {rv:7.1f}")

    # 7) Persist v2 artifacts
    joblib.dump(comb_pipe, MODEL_DIR / "lol_pipeline_v2.joblib")
    joblib.dump(combined_cols, MODEL_DIR / "feature_cols_v2.joblib")
    print(f"\n[save] {MODEL_DIR / 'lol_pipeline_v2.joblib'}  (+ feature_cols_v2.joblib)")
    print("[note] inference needs rating_diff, so predict.py must build the rating model too.")


if __name__ == "__main__":
    main()
