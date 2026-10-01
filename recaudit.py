"""recaudit - what accuracy scores do not tell you about a recommender.

A recommender can score well on precision, NDCG or RMSE while sending almost
everyone to the same small set of items. Standard evaluation does not surface
that, because every common metric is computed per user and then averaged, which
makes population-level behaviour invisible by construction.

This module takes any recommender and reports four things alongside accuracy:

  coverage         what share of the catalogue is ever recommended to anyone
  popularity bias  how popular recommended items are, against the catalogue
  personalisation  how much two random users' lists actually differ
  history effect   whether personalisation improves as a user's history grows

Usage
-----
    from recaudit import audit

    report = audit(
        recommend=lambda user, n: my_model.top_n(user, n),
        train=train_df,          # userId, movieId, rating
        test=test_df,
        n=10,
    )
    print(report)

The only thing required of a model is a function that returns the top n item
ids for a user, so this works with any library or a hand-written model.
"""
from itertools import combinations

import numpy as np
import pandas as pd

__all__ = ["audit", "AuditReport"]


class AuditReport(dict):
    """A dict that prints as a readable report."""

    def __str__(self):
        lines = ["", "=" * 62, "  RECOMMENDER AUDIT", "=" * 62]

        lines.append("\n  ACCURACY")
        for k in ("precision", "ndcg", "recall"):
            if k in self:
                lines.append(f"    {k:<34s} {self[k]:.4f}")

        lines.append("\n  CONCENTRATION")
        lines.append(f"    catalogue coverage                 "
                     f"{self['catalogue_coverage']:.1%}")
        lines.append(f"    items filling half of all slots    "
                     f"{self['items_to_half_of_slots']}")
        lines.append(f"    gini of recommendation counts      "
                     f"{self['gini']:.3f}")

        lines.append("\n  POPULARITY BIAS")
        lines.append(f"    mean popularity percentile of recs "
                     f"{self['mean_popularity_percentile']:.3f}")
        lines.append(f"    catalogue reference                "
                     f"{self['catalogue_popularity_reference']:.3f}")
        verdict = ("above catalogue average"
                   if self["mean_popularity_percentile"] >
                   self["catalogue_popularity_reference"] else "at or below")
        lines.append(f"    verdict                            {verdict}")

        lines.append("\n  PERSONALISATION")
        lines.append(f"    mean shared items between 2 users  "
                     f"{self['mean_shared_items']:.2f} of {self['n']}")
        lines.append(f"    popularity-only reference          "
                     f"{self['popularity_reference_shared']:.2f} of {self['n']}")
        if self["mean_shared_items"] >= self["popularity_reference_shared"]:
            lines.append("    WARNING: lists overlap as much as or more than "
                         "ranking by popularity alone.")
            lines.append("             This model may not be personalising.")
        lines.append(f"    most recommended item reaches      "
                     f"{self['top_item_reach']:.1%} of users")

        if self.get("history_effect"):
            lines.append("\n  PERSONALISATION BY USER HISTORY")
            for bucket, val in self["history_effect"].items():
                lines.append(f"    {bucket:<34s} {val:.2f} shared")

        lines.append("\n" + "=" * 62)
        return "\n".join(lines)


def _gini(counts):
    x = np.sort(np.asarray(counts, dtype=float))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((2 * (idx * x).sum()) / (n * x.sum()) - (n + 1) / n)


def _items_to_half(counts):
    x = np.sort(np.asarray(counts, dtype=float))[::-1]
    if x.sum() == 0:
        return 0
    return int(np.searchsorted(np.cumsum(x) / x.sum(), 0.5) + 1)


def audit(recommend, train, test=None, n=10, relevant_threshold=4.0,
          user_col="userId", item_col="movieId", rating_col="rating",
          max_pairs=20000, seed=42, history_buckets=4):
    """Audit a recommender.

    Parameters
    ----------
    recommend : callable(user_id, n) -> sequence of item ids
    train     : DataFrame of observed interactions
    test      : optional held-out DataFrame, needed for accuracy metrics
    n         : list length to audit
    """
    rng = np.random.default_rng(seed)
    users = np.sort(train[user_col].unique())
    catalogue = np.sort(train[item_col].unique())

    pop = train.groupby(item_col).size()
    pop_pct = pop.rank(pct=True)
    catalogue_reference = float(pop_pct.mean())

    lists = {}
    for u in users:
        try:
            recs = np.asarray(list(recommend(u, n)))
        except Exception:
            continue
        if len(recs):
            lists[u] = recs
    if not lists:
        raise ValueError("recommend() returned nothing for every user")

    flat = [i for v in lists.values() for i in v]
    counts = pd.Series(flat).value_counts()
    full = pd.Series(0.0, index=catalogue)
    common = counts.index.intersection(full.index)
    full.loc[common] = counts.loc[common].to_numpy()

    report = AuditReport({
        "n": n,
        "n_users_audited": len(lists),
        "catalogue_size": len(catalogue),
        "catalogue_coverage": float(len(set(flat)) / len(catalogue)),
        "items_to_half_of_slots": _items_to_half(full.to_numpy()),
        "gini": _gini(full.to_numpy()),
        "mean_popularity_percentile": float(np.mean(
            [pop_pct.reindex(v).fillna(0).mean() for v in lists.values()])),
        "catalogue_popularity_reference": catalogue_reference,
        "top_item_reach": float(counts.iloc[0] / len(lists)),
    })

    # ---- personalisation, against a popularity-only reference
    def _mean_shared(subset):
        pairs = list(combinations(subset, 2))
        if not pairs:
            return 0.0
        if len(pairs) > max_pairs:
            idx = rng.choice(len(pairs), size=max_pairs, replace=False)
            pairs = [pairs[i] for i in idx]
        return float(np.mean([len(set(lists[a]) & set(lists[b]))
                              for a, b in pairs]))

    report["mean_shared_items"] = _mean_shared(list(lists.keys()))

    seen = train.groupby(user_col)[item_col].apply(set).to_dict()
    pop_order = pop.sort_values(ascending=False).index.to_numpy()
    pop_lists = {}
    for u in lists:
        s = seen.get(u, set())
        pop_lists[u] = np.array([i for i in pop_order if i not in s][:n])
    saved, lists = lists, pop_lists
    report["popularity_reference_shared"] = _mean_shared(list(lists.keys()))
    lists = saved

    # ---- does personalisation improve with history?
    hist = train.groupby(user_col).size()
    try:
        buckets = pd.qcut(hist, history_buckets, duplicates="drop")
        effect = {}
        for b, grp in hist.groupby(buckets, observed=True):
            subset = [u for u in grp.index if u in lists]
            if len(subset) > 1:
                lo, hi = int(grp.min()), int(grp.max())
                effect[f"{lo}-{hi} ratings"] = _mean_shared(subset)
        report["history_effect"] = effect
    except ValueError:
        report["history_effect"] = {}

    # ---- accuracy, if a test set was supplied
    if test is not None:
        liked = test[test[rating_col] >= relevant_threshold]
        rel = liked.groupby(user_col)[item_col].apply(set).to_dict()
        precisions, ndcgs, recalls = [], [], []
        for u, relevant in rel.items():
            if u not in lists or not relevant:
                continue
            hits = [1.0 if i in relevant else 0.0 for i in lists[u]]
            precisions.append(sum(hits) / n)
            recalls.append(sum(hits) / len(relevant))
            dcg = sum(h / np.log2(r + 2) for r, h in enumerate(hits))
            ideal = sum(1.0 / np.log2(r + 2)
                        for r in range(min(len(relevant), n)))
            ndcgs.append(dcg / ideal if ideal > 0 else 0.0)
        if precisions:
            report["precision"] = float(np.mean(precisions))
            report["recall"] = float(np.mean(recalls))
            report["ndcg"] = float(np.mean(ndcgs))

    return report
