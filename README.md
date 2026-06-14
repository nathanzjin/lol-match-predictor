---
title: LoL Match Predictor
emoji: 🎮
colorFrom: blue
colorTo: red
sdk: docker
app_port: 8000
pinned: false
---

# lol-match-predictor

Predicts the outcome of professional League of Legends matches from the **players
on each team**, trained on [Oracle's Elixir](https://oracleselixir.com/) data. It
ships as a small web app: pick any two Tier-1 teams for a live win-probability,
and watch the model's accuracy graded against real results as the season unfolds.

Two parts, covered below: **how the model was trained**, and **the web app**.

---

## How the model was trained

**Data.** Oracle's Elixir match data, 2023–2026 (one row per player and per team,
per game). It's large and reproducible, so it's not in git — `download_data.py`
fetches it into `data/raw/`. Everything is **leakage-safe**: games are ordered by
date with a temporal train/test split, and all ratings are computed online
(predict a game from pre-game state, *then* update).

**The approach evolved in stages** (each kept in the repo as a baseline):

1. **Team rolling form** (`train_v1.py`) — XGBoost / logistic regression over a
   team's last-10-games form. ~60–62% accuracy.
2. **Team Elo** (`elo.py`) — a simple opponent-strength rating that beat the form
   model on its own.
3. **Region-anchored Elo** (`region_elo.py`) — splits strength into within-region
   skill + region strength, fixing cross-region (international) prediction.
4. **Player-level model** (`train_v3.py`) — the current best. Detailed below.

**The shipped model** treats a team as its five players (rosters move — transfers,
subs, off-season swaps — and player signals update the instant a lineup changes):

- **Player Elo that travels with the player** — one rating per player; a team's
  strength is the mean of its five. New lineups are rated correctly from game one.
- **Region-anchored player Elo** — within-region skill + region strength, so a
  player farming a weak league can't out-rate real stars, and cross-region games
  become a real signal. Region strength is a slow latent (small `k_region`, tuned
  to match the all-time head-to-head record), giving the order
  **KR > CN > EMEA > NA > BR > APAC**.
- **Per-role recent form** — each player's recent `dpm`, `cspm`, `vspm`,
  `earned gpm`, `damageshare`, KDA as per-role differentials. These exist on
  *partial* data rows too, so LPL and international games are covered.
- **Roster continuity** — how many starters carried over from the previous game.

These are combined in XGBoost ("combined-both"). **Scope:** predictions are
supported for Riot's six Tier-1 regions — **LCK (KR), LPL (CN), LEC (EMEA),
LCS (NA), CBLOL (BR), LCP (APAC)** (the 2025 LTA rebrands are folded back so a
team's region is stable across seasons). Every other league is kept as training
**breadth** but is never a prediction target.

**Results** (temporal holdout, ~7.4k unseen games):

| Model | Accuracy | Log-loss | ROC-AUC |
|-------|:--------:|:--------:|:-------:|
| team Elo | 0.630 | 0.646 | 0.679 |
| region-anchored Elo | 0.632 | 0.644 | 0.682 |
| player Elo | 0.644 | 0.631 | 0.698 |
| **player-level (combined-both)** | **0.659** | **0.615** | **0.719** |

Going player-level beats team-level by a statistically significant margin
(McNemar p=0.005), helps most exactly when a lineup just changed, and is the best
the project has produced. `backtest_players.py` runs the full comparison with
paired confidence intervals and significance tests.

```bash
pip install -r requirements.txt
python download_data.py     # fetch Oracle's Elixir CSVs -> data/raw/
python train_v3.py          # train + save the model to models/
python predict.py "T1" "Gen.G"                      # one-off prediction (CLI)
python predict.py "T1" "Gen.G" --blue-roster mid=Faker   # try a roster change
```

---

## The web app

A FastAPI backend (`api.py`) wraps the prediction core (`predictor.py`) and serves
a static frontend (`static/`). Run it locally:

```bash
uvicorn api:app --host 0.0.0.0 --port 8000     # open http://localhost:8000
```

**Two tabs:**

1. **Predict a matchup** — pick any two Tier-1 teams (grouped by region) and get a
   win probability, both rosters with player Elos, and the key signals behind the
   call.
2. **Season tracker** — the model's *walk-forward* track record for the current
   season (`season.py`): train on everything through the previous year, then grade
   the model on every unseen Tier-1 matchup. Shows a cumulative accuracy / log-loss
   chart, per-league splits, and a recent-results log. The record extends as new
   match data is downloaded through the year.

**API endpoints** (JSON under `/api`):

| Endpoint | Description |
|---|---|
| `GET /api/teams` | Supported Tier-1 teams grouped by region |
| `GET /api/teams/{team}/roster` | A team's most-recent roster |
| `POST /api/predict` | `{blue, red, window?, blue_roster?, red_roster?}` → prediction |
| `GET /api/performance` | The model's walk-forward season track record |

The season computation replays the full history once and runs in a background
thread on startup; `/api/performance` reports `computing` until it's ready
(~30–60s on first load), then serves the cached result.

---

## Deploying it

The app isn't a lightweight stateless function: it loads ~290 MB of match data
into memory and runs a ~minute-long computation for the season tracker. So it
wants a **persistent container with ≥2 GB RAM**, not a serverless platform.

A `Dockerfile` is included that bootstraps everything (downloads the data and
trains the model at build time, then serves on `$PORT`):

```bash
docker build -t lol-predictor .
docker run -p 8000:8000 lol-predictor
```

**Recommended hosts** (all run a container directly): **Render**, **Railway**,
**Fly.io**, or **Hugging Face Spaces** (Docker SDK). Point them at this repo /
Dockerfile, give the instance ≥2 GB RAM, and they'll build and serve it. Rebuild
to refresh the data and extend the season track record.

### Hugging Face Spaces (free)

The YAML header at the top of this README configures a Docker Space
(`sdk: docker`, `app_port: 8000`). The free **CPU basic** hardware has enough RAM
for the in-memory data (Render's free 512 MB tier does not).

1. Create a new Space at <https://huggingface.co/new-space> — **SDK: Docker**,
   hardware **CPU basic (free)**.
2. Push this repo to the Space's git remote (the data and model aren't committed;
   the Dockerfile downloads + trains them during the build):
   ```bash
   git remote add space https://huggingface.co/spaces/<you>/<space-name>
   git push space HEAD:main
   ```
3. The first build takes a few minutes (download ~290 MB + train). Once it's up,
   the Season tab needs ~30–60s on first load while it replays history, then it's
   cached.

**About Vercel:** Vercel is great for the *static frontend* but a poor fit for
this *backend* — serverless functions have tight size, memory, and execution-time
limits and cold-start fresh, which clashes with loading 290 MB of data and a
minute-long season computation. If you specifically want Vercel, the clean split
is: host the `static/` frontend on Vercel and point it at the FastAPI API running
on one of the container hosts above (set the API base URL in `static/app.js`).

---

## Repo layout

```
.
├── download_data.py       # fetch Oracle's Elixir data -> data/raw/ (git-ignored)
├── player_features.py     # player Elo, region-anchored player Elo, per-role stats, rosters
├── train_v3.py            # train + save the player-level model -> models/ (git-ignored)
├── backtest_players.py    # player-level vs team-level comparison + significance tests
├── predictor.py           # reusable prediction core (used by the CLI and the API)
├── predict.py             # CLI: predict a matchup from recent rosters
├── season.py              # walk-forward season track record
├── api.py                 # FastAPI service (predict + performance endpoints)
├── static/                # web UI (matchup picker + season tracker)
├── Dockerfile             # build = fetch data + train; run = serve uvicorn
├── train_v1.py            # baseline: team rolling-form model
├── elo.py / region_elo.py # baseline rating systems (team / region-anchored)
└── backtest*.py           # evaluation harnesses + shared metrics
```

## Roadmap

- [x] Player-level model: player Elo (region-anchored) + per-role stats + roster association
- [x] Tier-1 region scope; minor leagues used as training breadth only
- [x] FastAPI service + web UI (matchup picker + season tracker)
- [ ] Head-to-head feature (historical win rate between the two teams)
- [ ] Leakage-safe draft / champion win-rate features
- [ ] Hyperparameter tuning (Optuna + TimeSeriesSplit)

## Data attribution

Match data courtesy of [Oracle's Elixir](https://oracleselixir.com/). Free for
non-commercial use with attribution.
