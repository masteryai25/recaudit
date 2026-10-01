"""Experiment 3 - what happens when the system trains on its own output?

A deployed recommender does not sit still. It recommends, people mostly watch
what is put in front of them, and those choices become the next training set.
We simulate that loop and measure whether the range of content narrows.

The simulated user is deliberately simple and its assumptions are stated, since
this is the part of the project that is not validated against real behaviour:
  - a user considers the top n recommendations plus some organic discovery
  - the chance they pick an item rises with the model's predicted rating
  - the rating they give is the model's ground-truth-like preference plus noise

Two controls make the result interpretable:
  'no_feedback'  - same simulated choices, but the model is never retrained
  'random_recs'  - recommendations chosen at random instead of by the model
"""
import json

N_EPOCHS_LOOP = 12

import numpy as np
import pandas as pd

from data import load_ratings, temporal_split
from models import Baseline, UserCF, MatrixFactorisation


def gini(counts):
    """0 = every item consumed equally, 1 = all consumption on one item."""
    x = np.sort(np.asarray(counts, dtype=float))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((2 * (idx * x).sum()) / (n * x.sum()) - (n + 1) / n)


def effective_catalogue(counts, frac=0.8):
    """How many distinct items account for `frac` of all consumption."""
    x = np.sort(np.asarray(counts, dtype=float))[::-1]
    if x.sum() == 0:
        return 0
    return int(np.searchsorted(np.cumsum(x) / x.sum(), frac) + 1)


class SimulatedUsers:
    """Latent preferences for each user, fitted once on the real data.

    These stand in for 'what the user would actually think of this film'. They
    come from a matrix factorisation fitted to the full real dataset, so the
    preferences are grounded in real behaviour even though the choices are not.
    """

    def __init__(self, ratings, seed=42):
        self.rng = np.random.default_rng(seed)
        self.truth = MatrixFactorisation(n_factors=32, n_epochs=30,
                                         seed=seed).fit(ratings)

    def true_rating(self, user, items):
        items = np.asarray(items)
        scores, all_items = self.truth.score_all(user)
        lookup = dict(zip(all_items.tolist(), scores.tolist()))
        return np.array([lookup.get(i, self.truth.mu) for i in items.tolist()])

    def choose(self, user, recommended, catalogue, n_choices=3,
               organic_frac=0.2, temperature=1.0):
        """Pick items to consume: mostly from recommendations, some organic."""
        n_organic = int(round(n_choices * organic_frac))
        n_from_recs = n_choices - n_organic

        chosen = []
        if n_from_recs > 0 and len(recommended) > 0:
            scores = self.true_rating(user, recommended)
            w = np.exp((scores - scores.max()) / temperature)
            w = w / w.sum()
            k = min(n_from_recs, len(recommended))
            chosen += list(self.rng.choice(recommended, size=k,
                                           replace=False, p=w))
        if n_organic > 0:
            pool = np.setdiff1d(catalogue, np.array(chosen))
            if len(pool) > 0:
                k = min(n_organic, len(pool))
                chosen += list(self.rng.choice(pool, size=k, replace=False))

        items = np.array(chosen)
        ratings = self.true_rating(user, items) + self.rng.normal(0, 0.3,
                                                                 len(items))
        return items, np.clip(ratings, 0.5, 5.0)


def run_loop(condition, train, sim, n_rounds=10, n_recs=10, n_choices=3,
             seed=42):
    """Run one condition of the simulation for n_rounds."""
    rng = np.random.default_rng(seed)
    data = train.copy()
    catalogue = np.sort(train.movieId.unique())
    users = np.sort(train.userId.unique())

    consumption = pd.Series(0, index=catalogue, dtype=float)
    history = []

    model = MatrixFactorisation(n_factors=32, n_epochs=N_EPOCHS_LOOP, seed=seed).fit(data)

    for rnd in range(n_rounds):
        round_counts = pd.Series(0, index=catalogue, dtype=float)
        seen = data.groupby("userId").movieId.apply(set).to_dict()

        for user in users:
            unseen = np.setdiff1d(catalogue, np.array(sorted(seen.get(user, set()))))
            if len(unseen) == 0:
                continue

            if condition == "random_recs":
                recs = rng.choice(unseen, size=min(n_recs, len(unseen)),
                                  replace=False)
            else:
                all_scores, all_items = model.score_all(user)
                keep = np.isin(all_items, unseen)
                cand, cs = all_items[keep], all_scores[keep]
                recs = cand[np.argsort(-cs)[:n_recs]]

            items, rates = sim.choose(user, recs, unseen, n_choices=n_choices)
            round_counts[items] += 1

            new = pd.DataFrame({"userId": user, "movieId": items,
                                "rating": rates, "timestamp": 0})
            data = pd.concat([data, new], ignore_index=True)

        consumption += round_counts
        history.append({
            "round": rnd + 1,
            "condition": condition,
            "round_gini": round(gini(round_counts.to_numpy()), 4),
            "round_effective_catalogue": effective_catalogue(round_counts.to_numpy()),
            "round_distinct_items": int((round_counts > 0).sum()),
            "gini": round(gini(consumption.to_numpy()), 4),
            "effective_catalogue_80pct": effective_catalogue(consumption.to_numpy()),
            "distinct_items_consumed": int((consumption > 0).sum()),
            "mean_rating_given": round(float(data.rating.tail(
                len(users) * n_choices).mean()), 3),
        })
        print(f"  round {rnd+1:2d}  round_gini {history[-1]['round_gini']:.4f}  "
              f"round_distinct {history[-1]['round_distinct_items']:5d}  "
              f"round_80pct {history[-1]['round_effective_catalogue']:5d}")
        pd.DataFrame(history).to_csv(f"fb_{condition}.csv", index=False)

        if condition != "no_feedback":
            model = MatrixFactorisation(n_factors=32, n_epochs=30,
                                        seed=seed).fit(data)

    return history


def run(n_rounds=8, conditions=None):
    ratings = load_ratings()
    train, _ = temporal_split(ratings)

    print("fitting simulated user preferences on the real data...")
    sim = SimulatedUsers(ratings)

    all_history = []
    for condition in (conditions or ["feedback", "no_feedback", "random_recs"]):
        print(f"\n=== condition: {condition}")
        all_history += run_loop(condition, train, sim, n_rounds=n_rounds)

    df = pd.DataFrame(all_history)
    df.to_csv("results_feedback_loop.csv", index=False)

    print("\n=== concentration after the final round ===")
    final = df[df["round"] == df["round"].max()]
    print(final[["condition", "gini", "effective_catalogue_80pct",
                 "distinct_items_consumed"]].to_string(index=False))

    with open("results_feedback_loop.json", "w") as f:
        json.dump(all_history, f, indent=2)
    return df


if __name__ == "__main__":
    import sys
    conds = sys.argv[1:] or None
    run(conditions=conds)
