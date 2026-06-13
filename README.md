# lol-match-predictor

A machine-learning model that predicts the outcome of professional League of Legends
esports matches, trained on [Oracle's Elixir](https://oracleselixir.com/) match data.

This is a **v1 / learning build**: one script, a small set of leakage-safe features,
and a baseline vs. gradient-boosted model comparison. Tuning and additional features
come later.

## What it does

For each game it builds **one row per match** (blue side vs. red side) using only
information available *before* the game starts:

- **Rolling team form** over the last 10 games (win rate, gold diff @15, kills,
  first-tower / first-dragon / first-baron rates, vision, avg DPM/CSM) — computed
  strictly from prior games so there is **no data leakage**.
- **Differential features** (blue minus red) for each rolling stat.
- **Patch** as a numeric feature.

Target: did the **blue side win** (`1`) or not (`0`).

## Results (v1)

Trained on 2023 → mid-2025, evaluated on a temporal holdout of ~4,875 unseen
games from mid-to-late 2025:

| Model                       | Accuracy | ROC-AUC | Log-loss |
|-----------------------------|:--------:|:-------:|:--------:|
| Logistic Regression (baseline) | **0.616** | **0.659** | 0.649 |
| XGBoost                     | 0.602    | 0.652   | 0.665    |

- **Baseline to beat:** always predicting blue wins ~53.2% of the time (the real
  blue-side advantage), so the model adds ~8 points over naive guessing.
- The numbers sitting in the realistic **60–66%** range (not inflated into the 70s)
  is the signal that there's no leakage.
- With only ~10 mostly-linear features, **logistic regression beats XGBoost** — the
  trees have no complex interactions to exploit yet. XGBoost should pull ahead once
  richer features (draft, head-to-head, role-level stats) are added.
- Most predictive feature by far: **`diff_roll_winrate`** (recent form).

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # optional
pip install -r requirements.txt
```

## Get the data

The Oracle's Elixir CSVs are distributed via Google Drive and are **not committed**
to this repo (they're large and reproducible). Fetch them into `data/raw/`:

```bash
python download_data.py
```

If the default Drive folder is rate-limited ("quota exceeded"), make your own copy
of the folder in Google Drive, share it as *anyone with the link*, and pass it:

```bash
python download_data.py --folder "https://drive.google.com/drive/folders/<your_id>"
```

## Train

```bash
python train_v1.py
```

This loads `data/raw/`, builds features, trains both models, prints metrics and
feature importances, and writes artifacts to `models/`:

- `lol_pipeline_v1.joblib` — the full fitted pipeline (impute → scale → model)
- `feature_cols.joblib` — feature order for inference
- `calibration_v1.png` — calibration curve on the test set

## Player-level model (v3)

v1 rates *teams* and v2 added a region-anchored team rating. But a team is just
its five players, and rosters move — transfers, subs, off-season swaps — so
team-level signals lag every lineup change. v3 goes to the **individual-player
level**:

- **Player Elo that travels with the player** — one rating per `playername`, so
  when a player changes orgs their skill goes with them and the new lineup is
  rated correctly from game one. A team's strength is the mean of its five
  players' ratings.
- **Region-anchored player Elo** — the same idea as `region_elo.py`, one level
  down: a player's strength splits into within-region skill (updated only on
  domestic games, so it stays zero-sum around the region average) and region
  strength (updated only on international games). This cures plain player Elo's
  *weak-region inflation* — where a player farming a soft league out-rates real
  stars — and turns cross-region prediction from a coin flip into a real signal.
- **Per-role rolling stats** — each player's recent `dpm`, `cspm`, `vspm`,
  `earned gpm`, `damageshare` and KDA, turned into per-role differentials
  (`diff_mid_dpm`, `diff_bot_vspm`, …). These metrics exist for *partial* rows
  too, so LPL and international games are covered (v1's `golddiffat15` features
  silently drop them).
- **Roster continuity** — how many starters carried over from a team's previous
  game, flagging lineups the team-level ratings haven't caught up with.
- **Region-anchored team rating** folded in as one more feature.

The shipped model ("combined-both") feeds XGBoost **both** player ratings —
plain player Elo for the overall edge and the region-anchored one for
cross-region robustness — plus the per-role stats and region team rating.

Roster association: `player_features.most_recent_roster(team)` returns the five
players a team most recently fielded per role — the "current roster" used to
predict a hypothetical matchup.

### Results (v3, temporal holdout, ~7.4k unseen games)

| Model | Accuracy | Log-loss | ROC-AUC |
|-------|:--------:|:--------:|:-------:|
| team Elo (team rating) | 0.630 | 0.646 | 0.679 |
| region-anchored Elo (v2 headline) | 0.632 | 0.644 | 0.682 |
| **player Elo** (travels w/ player) | 0.644 | 0.631 | 0.698 |
| region-anchored player Elo | 0.647 | 0.630 | 0.700 |
| player stats (per-role rolling) | 0.650 | 0.624 | 0.708 |
| **combined-both (v3)** | **0.663** | **0.615** | **0.720** |

- **Going player-level beats team-level**: player Elo tops team Elo by +1.4 pts
  accuracy (McNemar p=0.005), and it isn't even tuned (team Elo was), so the
  gap is conservative.
- The **combined-both** model is the best the project has produced, +3.1 pts
  over the v2 region-anchored headline. Plain `pelo_diff` is the single most
  important feature.
- The player-level edge is **largest exactly when a lineup just changed**
  (low roster continuity) — the case team ratings handle worst.
- **Cross-region (different major regions, n=135)**: plain player Elo is a coin
  flip there (acc 0.541, AUC 0.629); region-anchoring rescues it to acc 0.681 /
  AUC 0.742 (d_acc **+0.141**, McNemar p=0.011), matching the region-elo
  specialist. The anchored *effective* leaderboard recovers real stars
  (Chovy, Knight, 369, Peanut, Kiin, Doran…) and the right region order:
  **CN ≈ KR ≫ EMEA > Americas > APAC**.
- Caveat: region-anchoring only calibrates regions that actually play
  international games (the major regions). Players in minor/"OTHER" leagues
  never face them, so their ratings still can't be placed on a global scale.

```bash
python train_v3.py                       # fit + save models/lol_pipeline_v3.joblib
python backtest_players.py               # full comparison + significance tests
python predict_v3.py "T1" "Gen.G"        # predict from each team's recent roster
python predict_v3.py --roster "T1"       # show a team's most-recent roster
python predict_v3.py "T1" "Gen.G" --blue-roster mid=Faker   # override a player
```

## Project layout

```
.
├── data/raw/              # OE CSVs (git-ignored; via download_data.py)
├── models/                # trained artifacts (git-ignored; via train_v1.py)
├── download_data.py       # fetch OE data from Google Drive
├── train_v1.py            # end-to-end v1 pipeline (team rolling form)
├── elo.py                 # team Elo baseline
├── region_elo.py          # region-anchored Elo (v2)
├── train_v2.py            # form + region-rating combined model
├── player_features.py     # player Elo, per-role stats, roster association
├── backtest_players.py    # player-level vs team-level comparison + stats
├── train_v3.py            # fit + save the player-level model (project best)
├── predict_v3.py          # predict a matchup from recent rosters
└── requirements.txt
```

## Roadmap

- [ ] Head-to-head feature (historical win rate between the two teams)
- [x] Role-level differentials (e.g. `diff_mid_dpm`, `diff_bot_vspm`) — see v3
- [x] Player-level stats + roster association — see v3
- [x] Fix player-Elo weak-region inflation (region-anchor the player ratings) — see v3
- [ ] Leakage-safe draft/champion win-rate features
- [ ] Calibrate minor/"OTHER" regions (needs more inter-region games or a prior)
- [ ] Hyperparameter tuning (Optuna + TimeSeriesSplit) and region as a feature
- [ ] FastAPI `/predict` endpoint + simple web UI

## Data attribution

Match data courtesy of [Oracle's Elixir](https://oracleselixir.com/). Free for
non-commercial use with attribution.
