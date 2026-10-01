"""Benchmark against established open-source recommenders.

Claiming a model is 'as good as what is out there' only means something if it is
measured against what is out there, on the same protocol. We use the `implicit`
library's ALS and BPR, both standard and widely used baselines.

Note on the task shift: ALS and BPR are implicit-feedback models. They do not
predict a rating, they rank items. That is the task we actually care about, so
from here on everything is evaluated as ranking. Explicit ratings are converted
to implicit signals in the usual way: a rating of 4 or more counts as a positive
interaction, everything else is dropped.

All models are judged on four things, not one:
    precision@10, ndcg@10   - is the list any good
    catalogue_coverage      - how much of the catalogue ever gets shown
    mean_shared_of_10       - how much two random users' lists overlap
"""
import json
import os
from itertools import combinations

import numpy as np
import pandas as pd
import scipy.sparse as sp

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from data import load_ratings, temporal_split

POSITIVE_THRESHOLD = 4.0
TOP_N = 10


def to_implicit(train, test, threshold=POSITIVE_THRESHOLD):
    """Keep only positive interactions and build a user-item matrix."""
    tr = train[train.rating >= threshold]
    te = test[test.rating >= threshold]

    users = np.sort(tr.userId.unique())
    items = np.sort(tr.movieId.unique())
    u_idx = {u: i for i, u in enumerate(users)}
    i_idx = {m: i for i, m in enumerate(items)}

    rows = tr.userId.map(u_idx).to_numpy()
    cols = tr.movieId.map(i_idx).to_numpy()
    mat = sp.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)),
                        shape=(len(users), len(items)))

    te = te[te.userId.isin(u_idx) & te.movieId.isin(i_idx)]
    truth = (te.assign(ui=te.userId.map(u_idx), ii=te.movieId.map(i_idx))
               .groupby("ui").ii.apply(set).to_dict())
    return mat, truth, users, items, u_idx, i_idx


def evaluate_ranking(recommend_fn, mat, truth, n_items, n=TOP_N,
                     n_pairs=20000, seed=42):
    """Score a model given a function returning top-n item indices for a user."""
    rng = np.random.default_rng(seed)
    precisions, ndcgs, lists = [], [], {}

    for ui, relevant in truth.items():
        if not relevant:
            continue
        recs = recommend_fn(ui, n)
        lists[ui] = recs
        hits = [1.0 if int(i) in relevant else 0.0 for i in recs]
        precisions.append(sum(hits) / n)
        dcg = sum(h / np.log2(r + 2) for r, h in enumerate(hits))
        ideal = sum(1.0 / np.log2(r + 2) for r in range(min(len(relevant), n)))
        ndcgs.append(dcg / ideal if ideal > 0 else 0.0)

    all_items = [i for v in lists.values() for i in v]
    coverage = len(set(all_items)) / n_items

    users = list(lists.keys())
    pairs = list(combinations(users, 2))
    if len(pairs) > n_pairs:
        idx = rng.choice(len(pairs), size=n_pairs, replace=False)
        pairs = [pairs[i] for i in idx]
    shared = float(np.mean([len(set(lists[a]) & set(lists[b]))
                            for a, b in pairs])) if pairs else 0.0

    counts = pd.Series(all_items).value_counts()
    return {
        f"precision@{n}": round(float(np.mean(precisions)), 4),
        f"ndcg@{n}": round(float(np.mean(ndcgs)), 4),
        "catalogue_coverage": round(coverage, 4),
        "mean_shared_of_10": round(shared, 3),
        "top10_items_share_of_slots": round(
            float(counts.head(10).sum() / len(all_items)), 4),
        "n_users_evaluated": len(precisions),
    }


def popularity_recommender(mat):
    pop = np.asarray(mat.sum(axis=0)).ravel()
    order = np.argsort(-pop)

    def rec(ui, n):
        seen = set(mat[ui].indices.tolist())
        out = [i for i in order if i not in seen][:n]
        return np.array(out)
    return rec


def run():
    ratings = load_ratings()
    train, test = temporal_split(ratings)
    mat, truth, users, items, u_idx, i_idx = to_implicit(train, test)
    n_items = mat.shape[1]
    print(f"implicit matrix: {mat.shape[0]} users x {n_items} items, "
          f"{mat.nnz:,} positive interactions")
    print(f"evaluating on {len(truth)} users with held-out positives\n")

    from implicit.als import AlternatingLeastSquares
    from implicit.bpr import BayesianPersonalizedRanking

    rows = []

    # -- reference: popularity only
    rec = popularity_recommender(mat)
    r = evaluate_ranking(rec, mat, truth, n_items)
    rows.append({"model": "Popularity (reference)", **r})
    print("Popularity (reference)", r)

    # -- established baselines
    for name, Model, kw in [
        ("ALS (implicit library)", AlternatingLeastSquares,
         dict(factors=64, regularization=0.05, iterations=20, random_state=42)),
        ("BPR (implicit library)", BayesianPersonalizedRanking,
         dict(factors=64, learning_rate=0.05, iterations=100, random_state=42)),
    ]:
        model = Model(**kw)
        model.fit(mat, show_progress=False)

        def rec(ui, n, _m=model):
            ids, _ = _m.recommend(ui, mat[ui], N=n,
                                  filter_already_liked_items=True)
            return np.asarray(ids)

        r = evaluate_ranking(rec, mat, truth, n_items)
        rows.append({"model": name, **r})
        print(name, r)

    df = pd.DataFrame(rows)
    df.to_csv("results_benchmark_baselines.csv", index=False)
    with open("results_benchmark_baselines.json", "w") as f:
        json.dump(rows, f, indent=2)

    print("\n=== baselines ===")
    print(df.to_string(index=False))
    return df


if __name__ == "__main__":
    run()
