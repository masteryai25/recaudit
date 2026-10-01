"""Experiment 6 - is the narrowing personalised, or popularity in disguise?

A recommender that genuinely personalises should send different people to
different parts of the catalogue. If it does that, any two users' lists should
overlap little, and the union of everyone's lists should cover a lot of the
catalogue.

Low coverage on its own does not settle this. Low coverage plus high overlap
between users means the system is largely recommending the same popular items
to everyone. Low coverage plus low overlap would mean something different:
real personalisation, but confined to a small region of the catalogue.

Reference points matter here, so we compare against:
  popularity_only - rank by how often an item was rated, ignoring the user
                    (maximum possible overlap: everyone gets the same list)
  random          - a random list per user (minimum overlap)
"""
import json
from itertools import combinations

import numpy as np
import pandas as pd

from data import load_ratings, temporal_split
from models import Baseline, UserCF, MatrixFactorisation


def jaccard(a, b):
    a, b = set(a), set(b)
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def top_lists(model, train, n=10, popularity=None, rng=None, mode="model"):
    """Top n for every user, excluding what they already rated."""
    items_all = np.sort(train.movieId.unique())
    seen = train.groupby("userId").movieId.apply(set).to_dict()
    lists = {}
    for user in np.sort(train.userId.unique()):
        mask = ~np.isin(items_all, np.array(sorted(seen.get(user, set()))))
        cand = items_all[mask]
        if len(cand) == 0:
            continue
        if mode == "random":
            lists[user] = rng.choice(cand, size=min(n, len(cand)),
                                     replace=False)
        elif mode == "popularity":
            p = popularity.reindex(cand).fillna(0).to_numpy()
            lists[user] = cand[np.argsort(-p)[:n]]
        else:
            if hasattr(model, "score_all"):
                scores_all, it = model.score_all(user)
                keep = np.isin(it, cand)
                c, s = it[keep], scores_all[keep]
                lists[user] = c[np.argsort(-s)[:n]]
            else:
                s = model.predict(np.full(len(cand), user), cand)
                lists[user] = cand[np.argsort(-s)[:n]]
    return lists


def overlap_stats(lists, n_pairs=20000, seed=42, n=10):
    rng = np.random.default_rng(seed)
    users = list(lists.keys())
    pairs = list(combinations(users, 2))
    if len(pairs) > n_pairs:
        idx = rng.choice(len(pairs), size=n_pairs, replace=False)
        pairs = [pairs[i] for i in idx]

    jac = [jaccard(lists[a], lists[b]) for a, b in pairs]
    shared = [len(set(lists[a]) & set(lists[b])) for a, b in pairs]

    all_items = [m for v in lists.values() for m in v]
    counts = pd.Series(all_items).value_counts()
    n_users = len(lists)

    return {
        "mean_jaccard": round(float(np.mean(jac)), 4),
        "mean_shared_items_of_10": round(float(np.mean(shared)), 3),
        "pairs_sharing_half_or_more": round(
            float(np.mean([s >= n / 2 for s in shared])), 4),
        "distinct_items_recommended": int(len(counts)),
        "most_recommended_item_reach": round(
            float(counts.iloc[0] / n_users), 4),
        "top10_items_share_of_all_slots": round(
            float(counts.head(10).sum() / len(all_items)), 4),
    }


def run(n=10):
    ratings = load_ratings()
    train, _ = temporal_split(ratings)
    popularity = train.groupby("movieId").size()
    rng = np.random.default_rng(42)

    configs = [
        ("random (reference: no personalisation, no popularity)", None, "random"),
        ("popularity only (reference: identical lists)", None, "popularity"),
        ("Baseline", Baseline().fit(train), "model"),
        ("UserCF", UserCF(k=40, shrinkage=10.0).fit(train), "model"),
        ("MatrixFactorisation",
         MatrixFactorisation(n_factors=32, n_epochs=30).fit(train), "model"),
    ]

    rows = []
    for name, model, mode in configs:
        lists = top_lists(model, train, n=n, popularity=popularity,
                          rng=rng, mode=mode)
        st = overlap_stats(lists, n=n)
        rows.append({"method": name, **st})
        print(f"{name}")
        print(f"   mean overlap between two users: "
              f"{st['mean_shared_items_of_10']:.2f} of {n} items "
              f"(jaccard {st['mean_jaccard']:.3f})")
        print(f"   pairs sharing half or more: "
              f"{st['pairs_sharing_half_or_more']:.1%}")
        print(f"   distinct items ever recommended: "
              f"{st['distinct_items_recommended']}")
        print(f"   single most recommended item reaches "
              f"{st['most_recommended_item_reach']:.1%} of users")
        print(f"   top 10 items fill "
              f"{st['top10_items_share_of_all_slots']:.1%} of all slots\n")

    df = pd.DataFrame(rows)
    df.to_csv("results_personalisation.csv", index=False)
    with open("results_personalisation.json", "w") as f:
        json.dump(rows, f, indent=2)
    return df


if __name__ == "__main__":
    run()
