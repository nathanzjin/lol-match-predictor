#!/usr/bin/env python3
"""
api.py - FastAPI service exposing the player-level LoL match predictor.

Endpoints (JSON under /api), plus the static frontend served at /:

  GET  /api/meta                  - regions, season info, predictor readiness
  GET  /api/teams                 - supported Tier-1 teams grouped by region
  GET  /api/teams/{team}/roster   - a team's most-recent roster (per role)
  POST /api/predict               - predict a hypothetical matchup
  GET  /api/performance           - the model's walk-forward season track record

The Predictor (arbitrary matchups) loads quickly and is ready at startup. The
season track record is heavier (it rebuilds the whole feature stream), so it's
computed once in a background thread; /api/performance reports "computing" until
it's ready, then serves the cached result.

Run:  uvicorn api:app --host 0.0.0.0 --port 8000
      (or: python api.py)
"""
from __future__ import annotations

import threading
import traceback
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from player_features import MAJOR_REGIONS
from predictor import Predictor, PredictError

STATIC_DIR = Path(__file__).parent / "static"

# Shared state populated at startup
STATE: dict = {
    "predictor": None,
    "predictor_error": None,
    "performance": None,        # cached season track record
    "performance_status": "idle",  # idle | computing | ready | error
    "performance_error": None,
}


def _compute_performance() -> None:
    STATE["performance_status"] = "computing"
    try:
        from season import compute_season_performance
        STATE["performance"] = compute_season_performance()
        STATE["performance_status"] = "ready"
    except Exception as e:  # surface, don't crash the server
        STATE["performance_error"] = str(e)
        STATE["performance_status"] = "error"
        traceback.print_exc()


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        STATE["predictor"] = Predictor()
    except Exception as e:
        STATE["predictor_error"] = str(e)
        traceback.print_exc()
    # Kick off the heavy season computation in the background
    threading.Thread(target=_compute_performance, daemon=True).start()
    yield


app = FastAPI(title="LoL Match Predictor", version="3.0", lifespan=lifespan)


def _predictor() -> Predictor:
    if STATE["predictor"] is None:
        raise HTTPException(503, STATE["predictor_error"] or "Predictor not ready.")
    return STATE["predictor"]


class PredictRequest(BaseModel):
    blue: str
    red: str
    window: int = 10
    blue_roster: dict[str, str] | None = None
    red_roster: dict[str, str] | None = None


@app.get("/api/meta")
def meta():
    p = STATE["predictor"]
    return {
        "regions": list(MAJOR_REGIONS),
        "predictor_ready": p is not None,
        "predictor_error": STATE["predictor_error"],
        "performance_status": STATE["performance_status"],
        "n_supported_teams": len(p.available) if p else 0,
    }


@app.get("/api/teams")
def teams():
    return _predictor().teams_by_region()


@app.get("/api/teams/{team}/roster")
def roster(team: str):
    p = _predictor()
    try:
        resolved = p.resolve(team)
    except PredictError as e:
        raise HTTPException(404, str(e))
    return {"team": resolved, "region": p.region_of.get(resolved),
            "roster": p.roster(resolved)}


@app.post("/api/predict")
def predict(req: PredictRequest):
    p = _predictor()
    if not 1 <= req.window <= 50:
        raise HTTPException(400, "window must be between 1 and 50.")
    try:
        return p.predict(req.blue, req.red, window=req.window,
                         blue_overrides=req.blue_roster, red_overrides=req.red_roster)
    except PredictError as e:
        raise HTTPException(400, str(e))


@app.get("/api/performance")
def performance():
    status = STATE["performance_status"]
    if status == "ready":
        return {"status": "ready", **STATE["performance"]}
    if status == "error":
        raise HTTPException(500, STATE["performance_error"] or "Performance computation failed.")
    return {"status": status}  # idle | computing


# Static frontend (mounted last so /api/* takes precedence)
if STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
