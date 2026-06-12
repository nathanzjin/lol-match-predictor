"""
v1 LoL esports match-outcome predictor.

Single-script pipeline:
  load -> clean -> aggregate players to team level -> build per-game rows
  -> no-leakage rolling features -> temporal split -> LogReg + XGBoost -> evaluate -> save.

Run:  python train_v1.py
"""
from __future__ import annotations
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless: save plots, never try to display
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, roc_auc_score, classification_report, log_loss,
)
from sklearn.calibration import CalibrationDisplay
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
DATA_DIR = Path("data/raw")
MODEL_DIR = Path("models")
YEARS = [2023, 2024, 2025]          # 2026 held out of training (partial, current season)
ROLLING_WINDOW = 10                 # games of history per team
MIN_GAMES = 5                       # need this many prior games or row is dropped
TRAIN_FRAC = 0.80                   # temporal split fraction
ROLES = ["top", "jng", "mid", "bot", "sup"]

# Columns we actually read (OE has 165; we need ~25). Note the SPACES in names.
USE_COLS = [
    "gameid", "date", "patch", "side", "teamname", "result", "gamelength",
    "league", "datacompleteness", "position",
    # player-level (aggregated to team)
    "dpm", "cspm", "vspm", "earned gpm",
    # team-level objectives / economy / combat / vision
    "golddiffat15", "kills", "firsttower", "firstdragon", "firstbaron", "visionscore",
]

# team-row columns we roll over -> output rolling column name
ROLL_MAP = {
    "result":       "roll_winrate",
    "golddiffat15": "roll_gd15",
    "kills":        "roll_kills",
    "firsttower":   "roll_firsttower",
    "firstdragon":  "roll_firstdragon",
    "firstbaron":   "roll_firstbaron",
    "visionscore":  "roll_vision",
    "avg_dpm":      "roll_dpm",
    "avg_cspm":     "roll_cspm",
}


# --------------------------------------------------------------------------- #
# 1. Load + clean
# --------------------------------------------------------------------------- #
def load_and_clean() -> pd.DataFrame:
    frames = []
    for y in YEARS:
        path = DATA_DIR / f"{y}_LoL_esports_match_data_from_OraclesElixir.csv"
        frames.append(pd.read_csv(path, usecols=USE_COLS, low_memory=False))
    df = pd.concat(frames, ignore_index=True)
    print(f"[load] raw rows: {len(df):,}")

    # Keep only fully-tracked games (drops 'partial' rows missing in-game stats)
    df = df[df["datacompleteness"] == "complete"]
    # Valid result + sane game length (drops remakes/forfeits)
    df = df[df["result"].isin([0, 1])]
    df = df[df["gamelength"] > 900]
    # Types
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["patch"] = pd.to_numeric(df["patch"], errors="coerce")
    df = df.dropna(subset=["date", "gameid", "teamname", "side"])
    print(f"[clean] rows after filters: {len(df):,} | games: {df['gameid'].nunique():,}")
    return df


# --------------------------------------------------------------------------- #
# 2. Aggregate player rows -> team level, attach to team rows
# --------------------------------------------------------------------------- #
def build_team_frame(df: pd.DataFrame) -> pd.DataFrame:
    players = df[df["position"].isin(ROLES)]
    teams = df[df["position"] == "team"].copy()

    agg = (
        players.groupby(["gameid", "teamname"])
        .agg(
            avg_dpm=("dpm", "mean"),
            avg_cspm=("cspm", "mean"),
            avg_vspm=("vspm", "mean"),
            avg_egpm=("earned gpm", "mean"),
        )
        .reset_index()
    )
    teams = teams.merge(agg, on=["gameid", "teamname"], how="left")

    # keep exactly 2 team rows per game (1 blue, 1 red)
    good = teams.groupby("gameid")["side"].transform("nunique").eq(2)
    teams = teams[good]
    print(f"[team] team rows: {len(teams):,} | games: {teams['gameid'].nunique():,}")
    return teams


# --------------------------------------------------------------------------- #
# 3. No-leakage rolling features (only games strictly before the current one)
# --------------------------------------------------------------------------- #
def add_rolling(teams: pd.DataFrame) -> pd.DataFrame:
    teams = teams.sort_values(["date", "gameid"]).reset_index(drop=True)
    grp = teams.groupby("teamname", sort=False)
    for src, out in ROLL_MAP.items():
        # shift(1) => current game excluded => no leakage
        teams[out] = grp[src].transform(
            lambda s: s.shift(1).rolling(ROLLING_WINDOW, min_periods=MIN_GAMES).mean()
        )
    return teams


# --------------------------------------------------------------------------- #
# 4. Pivot to one row per game (blue vs red) + differentials
# --------------------------------------------------------------------------- #
def build_games(teams: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    roll_out = list(ROLL_MAP.values())
    base = ["gameid", "date", "patch", "teamname", "result"] + roll_out

    blue = teams.loc[teams["side"] == "Blue", base].rename(
        columns={"result": "blue_result", "teamname": "blue_teamname",
                 **{c: f"blue_{c}" for c in roll_out}}
    )
    red = teams.loc[teams["side"] == "Red", ["gameid", "teamname", "result"] + roll_out].rename(
        columns={"result": "red_result", "teamname": "red_teamname",
                 **{c: f"red_{c}" for c in roll_out}}
    )
    games = blue.merge(red, on="gameid", how="inner")
    games["target"] = games["blue_result"].astype(int)  # 1 = blue side won

    # differential features (blue - red): the strongest signal
    diff_feats = []
    for c in roll_out:
        name = f"diff_{c}"
        games[name] = games[f"blue_{c}"] - games[f"red_{c}"]
        diff_feats.append(name)

    features = diff_feats + ["patch"]

    before = len(games)
    games = games.dropna(subset=diff_feats)  # drop cold-start games (too little history)
    print(f"[games] usable games: {len(games):,} (dropped {before - len(games):,} cold-start)")
    return games.sort_values(["date", "gameid"]).reset_index(drop=True), features


# --------------------------------------------------------------------------- #
# 5. Temporal split + train + evaluate
# --------------------------------------------------------------------------- #
def evaluate(name, pipe, X_tr, y_tr, X_te, y_te):
    pipe.fit(X_tr, y_tr)
    proba = pipe.predict_proba(X_te)[:, 1]
    pred = (proba >= 0.5).astype(int)
    acc = accuracy_score(y_te, pred)
    auc = roc_auc_score(y_te, proba)
    ll = log_loss(y_te, proba)
    print(f"\n=== {name} ===")
    print(f"  accuracy : {acc:.4f}")
    print(f"  roc_auc  : {auc:.4f}")
    print(f"  log_loss : {ll:.4f}")
    print(classification_report(y_te, pred, target_names=["red wins", "blue wins"], digits=3))
    return pipe, proba, {"accuracy": acc, "roc_auc": auc, "log_loss": ll}


def main():
    MODEL_DIR.mkdir(exist_ok=True)
    df = load_and_clean()
    teams = build_team_frame(df)
    teams = add_rolling(teams)
    games, features = build_games(teams)

    print(f"\n[features] {len(features)}: {features}")
    print(f"[target] blue-win base rate (side advantage): {games['target'].mean():.4f}")

    split = int(len(games) * TRAIN_FRAC)
    train, test = games.iloc[:split], games.iloc[split:]
    print(f"[split] train {len(train):,} ({train['date'].min().date()} -> {train['date'].max().date()}) | "
          f"test {len(test):,} ({test['date'].min().date()} -> {test['date'].max().date()})")

    X_tr, y_tr = train[features], train["target"]
    X_te, y_te = test[features], test["target"]

    # Baseline: logistic regression
    logreg = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000)),
    ])
    evaluate("Logistic Regression (baseline)", logreg, X_tr, y_tr, X_te, y_te)

    # Main model: XGBoost
    xgb = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", XGBClassifier(
            n_estimators=400, learning_rate=0.05, max_depth=4,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
            eval_metric="logloss", n_jobs=-1, random_state=42,
        )),
    ])
    xgb, proba, _ = evaluate("XGBoost", xgb, X_tr, y_tr, X_te, y_te)

    # Feature importances
    importances = xgb.named_steps["model"].feature_importances_
    print("\n[xgb feature importance]")
    for f, imp in sorted(zip(features, importances), key=lambda x: -x[1]):
        print(f"  {f:<22} {imp:.4f}")

    # Calibration curve
    fig, ax = plt.subplots(figsize=(6, 6))
    CalibrationDisplay.from_predictions(y_te, proba, n_bins=10, ax=ax, name="XGBoost")
    ax.set_title("Calibration - XGBoost (test set)")
    fig.savefig(MODEL_DIR / "calibration_v1.png", dpi=110, bbox_inches="tight")
    print(f"\n[plot] saved {MODEL_DIR / 'calibration_v1.png'}")

    # Persist artifacts
    joblib.dump(xgb, MODEL_DIR / "lol_pipeline_v1.joblib")
    joblib.dump(features, MODEL_DIR / "feature_cols.joblib")
    print(f"[save] {MODEL_DIR / 'lol_pipeline_v1.joblib'}")
    print(f"[save] {MODEL_DIR / 'feature_cols.joblib'}")


if __name__ == "__main__":
    main()
