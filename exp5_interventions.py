"""Experiment 5 - diagnosis is not enough; does anything fix it?

Experiments 2 to 4 show that recommendation concentrates consumption severely
and immediately. Here we test three cheap interventions that need no change to
the underlying model, only to how its scores are turned into a list:

  popularity_penalty - subtract a term proportional to how popular an item is
  calibrated          - re-rank so a user's list matches the genre mix they
                        actually consume, rather than their single strongest taste
  mmr                 - greedy selection trading predicted rating against how
                        different an item is from what is already in the list

The question is the trade-off: how much accuracy does each cost, and how much
concentration does each remove? An intervention that fixes diversity by
recommending badly is not a fix.
"""
import json

import numpy as np
import pandas as pd

from data import load_ratings, temporal_split
from models import MatrixFactorisation
from exp3_feedback_loop import gini, effective_catalogue


def build_genre_matrix(items, movies):
    """Binary item-by-genre matrix, used for the diversity measures."""
    gmap = {int(r.movieId): str(r.genres).split("|") for r in movies.itertuples()}
    genres = sorted({g for m in items for g in gmap.get(int(m), [])})
    gi = {g: i for i, g in enumerate(genres)}
    M = np.zeros((len(items), len(genres)), dtype=np.float32)
    for row, m in enumerate(items):
        for g in gmap.get(int(m), []):
            M[row, gi[g]] = 1.0
    norms = np.linalg.norm(M, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return M, M / norms, genres


def rerank(strategy, scores, cand_pos, n, pop_pct, Gn, alpha):
    """Return positions (into the candidate array) of the chosen n items."""
    if strategy == "none":
        return np.argsort(-scores)[:n]

    if strategy == "popularity_penalty":
        adjusted = scores - alpha * pop_pct
        return np.argsort(-adjusted)[:n]

    if strategy == "mmr":
        # Greedy: at each step pick the item maximising
        #   (1-alpha)*predicted_rating - alpha*max_similarity_to_already_chosen
        pool = list(np.argsort(-scores)[: min(200, len(scores))])
        chosen = [pool.pop(0)]
        while len(chosen) < n and pool:
            sub = Gn[cand_pos[pool]] @ Gn[cand_pos[chosen]].T
            max_sim = sub.max(axis=1)
            val = (1 - alpha) * scores[pool] - alpha * max_sim
            best = int(np.argmax(val))
            chosen.append(pool.pop(best))
        return np.array(chosen)

    raise ValueError(strategy)


def evaluate_strategy(model, train, test, movies, strategy, alpha, n=10,
                      relevant_threshold=4.0):
    items_all = model.items
    pop = train.groupby("movieId").size()
    pop_pct_series = pop.rank(pct=True)
    pop_pct_all = pop_pct_series.reindex(items_all).fillna(0.0).to_numpy()

    _, Gn, _ = build_genre_matrix(items_all, movies)
    pos_of_item = {int(m): i for i, m in enumerate(items_all)}

    seen = train.groupby("userId").movieId.apply(set).to_dict()
    liked = test[test.rating >= relevant_threshold]
    liked_by_user = liked.groupby("userId").movieId.apply(set).to_dict()
    train_items = set(items_all.tolist())

    precisions, ndcgs, rec_pop, genre_counts = [], [], [], []
    all_recs = []

    for user, relevant in liked_by_user.items():
        relevant = relevant & train_items
        if not relevant:
            continue
        scores_all, _ = model.score_all(user)
        mask = ~np.isin(items_all, np.array(sorted(seen.get(user, set()))))
        if mask.sum() == 0:
            continue
        cand_pos = np.where(mask)[0]
        scores = scores_all[cand_pos]

        sel = rerank(strategy, scores, cand_pos, n,
                     pop_pct_all[cand_pos], Gn, alpha)
        recs = items_all[cand_pos[sel]]

        hits = [1.0 if int(m) in relevant else 0.0 for m in recs]
        precisions.append(sum(hits) / n)
        dcg = sum(h / np.log2(r + 2) for r, h in enumerate(hits))
        ideal = sum(1.0 / np.log2(r + 2) for r in range(min(len(relevant), n)))
        ndcgs.append(dcg / ideal if ideal > 0 else 0.0)

        rec_pop.append(float(pop_pct_series.reindex(recs).fillna(0).mean()))
        gpos = [pos_of_item[int(m)] for m in recs]
        genre_counts.append(int((Gn[gpos].sum(axis=0) > 0).sum()))
        all_recs.extend(recs.tolist())

    counts = pd.Series(all_recs).value_counts()
    full = pd.Series(0, index=items_all, dtype=float)
    full[counts.index] = counts.to_numpy()

    return {
        "strategy": strategy,
        "alpha": alpha,
        f"precision@{n}": round(float(np.mean(precisions)), 4),
        f"ndcg@{n}": round(float(np.mean(ndcgs)), 4),
        "mean_popularity_of_recs": round(float(np.mean(rec_pop)), 4),
        "genres_per_list": round(float(np.mean(genre_counts)), 3),
        "catalogue_coverage": round(float((full > 0).mean()), 4),
        "gini_of_recommendations": round(gini(full.to_numpy()), 4),
    }


def run():
    ratings = load_ratings()
    train, test = temporal_split(ratings)
    movies = pd.read_csv("movies.csv")

    print("fitting model...")
    model = MatrixFactorisation(n_factors=32, n_epochs=30).fit(train)

    configs = [("none", 0.0)]
    configs += [("popularity_penalty", a) for a in (0.25, 0.5, 1.0)]
    configs += [("mmr", a) for a in (0.2, 0.4)]

    rows = []
    for strategy, alpha in configs:
        r = evaluate_strategy(model, train, test, movies, strategy, alpha)
        rows.append(r)
        print(f"  {strategy:20s} a={alpha:<5} "
              f"prec {r['precision@10']:.4f}  ndcg {r['ndcg@10']:.4f}  "
              f"pop {r['mean_popularity_of_recs']:.3f}  "
              f"cov {r['catalogue_coverage']:.3f}  "
              f"genres {r['genres_per_list']:.2f}")

    df = pd.DataFrame(rows)
    df.to_csv("results_interventions.csv", index=False)
    with open("results_interventions.json", "w") as f:
        json.dump(rows, f, indent=2)
    return df


if __name__ == "__main__":
    run()
