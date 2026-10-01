"""Evaluation for all models.

Three families of measure:
  rating accuracy  - RMSE, MAE
  ranking quality  - Precision@K, Recall@K, NDCG@K
  popularity bias  - how popular the recommended items are versus the catalogue
"""
import numpy as np
import pandas as pd


# ---------------------------------------------------------------- accuracy

def rmse(pred, actual):
    return float(np.sqrt(np.mean((pred - actual) ** 2)))


def mae(pred, actual):
    return float(np.mean(np.abs(pred - actual)))


# ---------------------------------------------------------------- ranking

def top_n_for_user(model, user, candidates, n=10):
    """Score every candidate item for one user and return the best n."""
    users = np.full(len(candidates), user)
    scores = model.predict(users, candidates)
    order = np.argsort(-scores)[:n]
    return candidates[order], scores[order]


def ranking_metrics(model, train, test, n=10, relevant_threshold=4.0, seed=42):
    """Precision, recall and NDCG at n, plus popularity of what gets recommended.

    For each user we rank every item they did not rate in training, and check
    how many of their genuinely liked test items (rating >= threshold) appear in
    the top n. This is the question a real system faces: out of everything
    available, what do you put in front of this person?
    """
    rng = np.random.default_rng(seed)
    train_items = set(train.movieId.unique())
    seen = train.groupby("userId").movieId.apply(set).to_dict()

    # popularity measured on training data only
    pop = train.groupby("movieId").size()
    pop_rank = pop.rank(pct=True)  # 1.0 = most popular item in the catalogue

    precisions, recalls, ndcgs, rec_pop = [], [], [], []

    liked = test[test.rating >= relevant_threshold]
    liked_by_user = liked.groupby("userId").movieId.apply(set).to_dict()

    for user, relevant in liked_by_user.items():
        relevant = relevant & train_items  # cannot recommend unseen items
        if not relevant:
            continue
        candidates = np.array(sorted(train_items - seen.get(user, set())))
        if len(candidates) == 0:
            continue

        recs, _ = top_n_for_user(model, user, candidates, n=n)
        hits = [1.0 if m in relevant else 0.0 for m in recs]

        precisions.append(sum(hits) / n)
        recalls.append(sum(hits) / len(relevant))

        dcg = sum(h / np.log2(rank + 2) for rank, h in enumerate(hits))
        ideal = sum(1.0 / np.log2(rank + 2)
                    for rank in range(min(len(relevant), n)))
        ndcgs.append(dcg / ideal if ideal > 0 else 0.0)

        rec_pop.append(float(pop_rank.reindex(recs).fillna(0.0).mean()))

    return {
        f"precision@{n}": float(np.mean(precisions)),
        f"recall@{n}": float(np.mean(recalls)),
        f"ndcg@{n}": float(np.mean(ndcgs)),
        "mean_popularity_of_recs": float(np.mean(rec_pop)),
        "n_users_evaluated": len(precisions),
    }


def catalogue_coverage(model, train, n=10, seed=42):
    """What fraction of the catalogue ever appears in anyone's top n."""
    train_items = set(train.movieId.unique())
    seen = train.groupby("userId").movieId.apply(set).to_dict()
    recommended = set()
    for user in train.userId.unique():
        candidates = np.array(sorted(train_items - seen.get(user, set())))
        if len(candidates) == 0:
            continue
        recs, _ = top_n_for_user(model, user, candidates, n=n)
        recommended.update(recs.tolist())
    return len(recommended) / len(train_items)


# ---------------------------------------------------------------- breakdown

def error_breakdown(model, train, test, n_bins=4):
    """Where does each model actually win or lose?

    A single average error hides the interesting part. We split test ratings by
    how much history the user has and by how popular the item is, because the
    usual claim that matrix factorisation 'wins' may only hold in dense regions.
    """
    user_counts = train.groupby("userId").size()
    item_counts = train.groupby("movieId").size()

    df = test.copy()
    df["user_history"] = df.userId.map(user_counts).fillna(0)
    df["item_popularity"] = df.movieId.map(item_counts).fillna(0)
    df["pred"] = model.predict(df.userId.to_numpy(), df.movieId.to_numpy())
    df["sq_err"] = (df.pred - df.rating) ** 2

    out = {}
    for col in ["user_history", "item_popularity"]:
        try:
            bins = pd.qcut(df[col], n_bins, duplicates="drop")
        except ValueError:
            continue
        grouped = df.groupby(bins, observed=True).sq_err.mean().pow(0.5)
        out[col] = {str(k): round(float(v), 4) for k, v in grouped.items()}
    return out
