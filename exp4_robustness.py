"""Experiment 4 - is the flat result robust, and does it hide per-user narrowing?

Experiment 3 found that concentration is high immediately and does not grow over
eight rounds. Three things could explain that away, so we test all three:

  1. eight rounds is too short          -> run twenty
  2. our reliance assumption saturates  -> sweep how much users follow the recs
  3. the population is stable but each  -> track diversity per user, not just
     individual narrows                    across the whole population

Population-level and individual-level concentration can move in opposite
directions, and it is the individual one that matches what people usually mean
by a filter bubble.
"""
import json
import sys

import numpy as np
import pandas as pd

from data import load_ratings, temporal_split
from models import MatrixFactorisation
from exp3_feedback_loop import gini, effective_catalogue, SimulatedUsers

N_EPOCHS_LOOP = 12


def user_diversity(items, genre_map):
    """How many distinct genres one user consumed in a round."""
    genres = set()
    for m in items:
        genres.update(genre_map.get(m, []))
    return len(genres)


def load_genre_map(path="movies.csv"):
    mv = pd.read_csv(path)
    return {int(r.movieId): str(r.genres).split("|")
            for r in mv.itertuples()}


def run_condition(organic_frac, train, sim, genre_map, n_rounds=20,
                  n_recs=10, n_choices=5, seed=42, retrain=True):
    """organic_frac is the share of choices made outside the recommendations."""
    rng = np.random.default_rng(seed)
    data = train.copy()
    catalogue = np.sort(train.movieId.unique())
    users = np.sort(train.userId.unique())

    consumption = pd.Series(0, index=catalogue, dtype=float)
    user_items = {u: set() for u in users}
    history = []

    model = MatrixFactorisation(n_factors=32, n_epochs=N_EPOCHS_LOOP,
                                seed=seed).fit(data)

    for rnd in range(n_rounds):
        round_counts = pd.Series(0, index=catalogue, dtype=float)
        seen = data.groupby("userId").movieId.apply(set).to_dict()
        per_user_genres, per_user_new = [], []

        for user in users:
            unseen = np.setdiff1d(catalogue,
                                  np.array(sorted(seen.get(user, set()))))
            if len(unseen) == 0:
                continue
            scores, items_all = model.score_all(user)
            keep = np.isin(items_all, unseen)
            cand, cs = items_all[keep], scores[keep]
            recs = cand[np.argsort(-cs)[:n_recs]]

            items, rates = sim.choose(user, recs, unseen,
                                      n_choices=n_choices,
                                      organic_frac=organic_frac)
            round_counts[items] += 1
            per_user_genres.append(user_diversity(items, genre_map))
            per_user_new.append(len(set(items.tolist()) - user_items[user]))
            user_items[user].update(items.tolist())

            new = pd.DataFrame({"userId": user, "movieId": items,
                                "rating": rates, "timestamp": 0})
            data = pd.concat([data, new], ignore_index=True)

        consumption += round_counts
        history.append({
            "round": rnd + 1,
            "organic_frac": organic_frac,
            "retrain": retrain,
            "round_gini": round(gini(round_counts.to_numpy()), 4),
            "round_distinct_items": int((round_counts > 0).sum()),
            "round_80pct_items": effective_catalogue(round_counts.to_numpy()),
            "mean_genres_per_user": round(float(np.mean(per_user_genres)), 3),
            "mean_new_items_per_user": round(float(np.mean(per_user_new)), 3),
            "cumulative_gini": round(gini(consumption.to_numpy()), 4),
        })
        print(f"  r{rnd+1:2d} gini {history[-1]['round_gini']:.4f} "
              f"distinct {history[-1]['round_distinct_items']:5d} "
              f"genres/user {history[-1]['mean_genres_per_user']:.2f}")
        sys.stdout.flush()
        pd.DataFrame(history).to_csv(
            f"exp4_organic{int(organic_frac*100)}.csv", index=False)

        if retrain:
            model = MatrixFactorisation(n_factors=32, n_epochs=N_EPOCHS_LOOP,
                                        seed=seed).fit(data)

    return history


if __name__ == "__main__":
    organic = float(sys.argv[1])
    rounds = int(sys.argv[2]) if len(sys.argv) > 2 else 20

    ratings = load_ratings()
    train, _ = temporal_split(ratings)
    genre_map = load_genre_map()
    print("fitting simulated preferences...")
    sim = SimulatedUsers(ratings)
    print(f"=== organic fraction {organic} ({int((1-organic)*100)}% of choices "
          f"come from recommendations), {rounds} rounds")
    run_condition(organic, train, sim, genre_map, n_rounds=rounds)
