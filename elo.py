#!/usr/bin/env python3
"""
elo.py - a simple pre-game Elo rating baseline for LoL match prediction.

Why Elo here: the v1 rolling-form features describe *how* a team has played but
not *who* they played. Elo encodes opponent strength directly by passing rating
points between winner and loser, so strength-of-schedule (and cross-region gaps)
fall out for free. This is the baseline the feature-based model has to beat.

Pre-game discipline: every prediction is made from ratings as they stand *before*
the game, and only then is the result used to update them. No leakage.

The model is data-agnostic: it works on plain sequences of
(blue_team, red_team, blue_won) ordered by date. The __main__ block wires it to
the project's data just as a sanity check (prints the current top teams).

Run:  python elo.py
"""
from __future__ import annotations
from dataclasses import dataclass, field

import numpy as np


@dataclass
class EloModel:
    base: float = 1500.0       # starting rating for an unseen team
    k: float = 24.0            # update step size
    home_adv: float = 20.0     # blue-side edge, in Elo points (~+0.03 win prob)
    scale: float = 400.0       # logistic scale (standard Elo)
    ratings: dict[str, float] = field(default_factory=dict)

    def rating(self, team: str) -> float:
        return self.ratings.get(team, self.base)

    def expect_blue(self, blue: str, red: str) -> float:
        """Pre-game probability that the blue-side team wins."""
        diff = (self.rating(blue) + self.home_adv) - self.rating(red)
        return 1.0 / (1.0 + 10.0 ** (-diff / self.scale))

    def update(self, blue: str, red: str, blue_won: float) -> float:
        """Predict (pre-game), then move ratings toward the observed result.

        Returns the pre-update blue-win probability (the honest prediction).
        """
        p = self.expect_blue(blue, red)
        delta = self.k * (float(blue_won) - p)   # zero-sum: blue gains what red loses
        self.ratings[blue] = self.rating(blue) + delta
        self.ratings[red] = self.rating(red) - delta
        return p

    def fit_predict(self, blue_teams, red_teams, outcomes) -> np.ndarray:
        """Stream games in order; return the pre-game blue-win prob for each."""
        preds = [self.update(b, r, y) for b, r, y in zip(blue_teams, red_teams, outcomes)]
        return np.asarray(preds, dtype=float)


def _log_loss(y, p, eps: float = 1e-15) -> float:
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def tune(blue_teams, red_teams, outcomes,
         ks=(12, 16, 20, 24, 28, 32, 40, 48, 64),
         homes=(0, 10, 20, 30, 40),
         base: float = 1500.0, scale: float = 400.0) -> dict:
    """Grid-search K and home advantage on a (training-only) game stream.

    Picks the params with the lowest log loss on the *same* stream. Because Elo
    predicts before it updates, scoring on the training stream is leakage-free.
    """
    best = None
    for k in ks:
        for h in homes:
            m = EloModel(base=base, k=float(k), home_adv=float(h), scale=scale)
            preds = m.fit_predict(blue_teams, red_teams, outcomes)
            ll = _log_loss(outcomes, preds)
            if best is None or ll < best["train_log_loss"]:
                best = {"k": float(k), "home_adv": float(h), "train_log_loss": ll}
    return best


# --------------------------------------------------------------------------- #
# Sanity check against the project data
# --------------------------------------------------------------------------- #
def _demo() -> None:
    import pandas as pd
    from train_v1 import load_and_clean, build_team_frame

    teams = build_team_frame(load_and_clean())
    cols = ["gameid", "date", "side", "teamname", "result"]
    t = teams[cols].copy()
    blue = (t[t["side"] == "Blue"]
            .rename(columns={"teamname": "blue_team", "result": "blue_result"}))
    red = (t[t["side"] == "Red"]
           .rename(columns={"teamname": "red_team", "result": "red_result"}))
    games = (blue[["gameid", "date", "blue_team", "blue_result"]]
             .merge(red[["gameid", "red_team"]], on="gameid")
             .sort_values(["date", "gameid"])
             .reset_index(drop=True))
    games["target"] = games["blue_result"].astype(int)

    best = tune(games["blue_team"], games["red_team"], games["target"])
    print(f"[elo] tuned on full stream: K={best['k']:.0f}, "
          f"home_adv={best['home_adv']:.0f}, log_loss={best['train_log_loss']:.4f}")

    model = EloModel(k=best["k"], home_adv=best["home_adv"])
    model.fit_predict(games["blue_team"], games["red_team"], games["target"])

    # Only rank teams with enough games so the leaderboard is meaningful.
    counts = pd.concat([games["blue_team"], games["red_team"]]).value_counts()
    ranked = sorted(
        ((tm, r) for tm, r in model.ratings.items() if counts.get(tm, 0) >= 30),
        key=lambda x: -x[1],
    )
    print("\n[elo] top 15 teams by final rating (>=30 games):")
    for tm, r in ranked[:15]:
        print(f"  {r:7.1f}  {tm}")


if __name__ == "__main__":
    _demo()
