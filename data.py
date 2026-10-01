"""Load MovieLens 100K and split it per user, most recent ratings held out.

A per-user temporal split is used rather than a random split: recommending is a
prediction about the future, so the test set should be the future.
"""
import io
import os
import urllib.request
import zipfile

import pandas as pd
import numpy as np

RANDOM_SEED = 42

ML_100K_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
ML_1M_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"


def download_movielens(force=False):
    """Fetch MovieLens 100K and write ratings.csv and movies.csv alongside this
    script. Called automatically if the files are missing."""
    if not force and os.path.exists("ratings.csv") and os.path.exists("movies.csv"):
        return
    print(f"downloading MovieLens from {ML_100K_URL} ...")
    with urllib.request.urlopen(ML_100K_URL) as resp:
        blob = resp.read()
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        for name in ("ratings.csv", "movies.csv"):
            with z.open(f"ml-latest-small/{name}") as f:
                pd.read_csv(f).to_csv(name, index=False)
    print("saved ratings.csv and movies.csv")


def download_movielens_1m(force=False):
    """Fetch MovieLens 1M as ratings_1m.csv (used for the replication)."""
    if not force and os.path.exists("ratings_1m.csv"):
        return
    print(f"downloading MovieLens 1M from {ML_1M_URL} ...")
    with urllib.request.urlopen(ML_1M_URL) as resp:
        blob = resp.read()
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        with z.open("ml-1m/ratings.dat") as f:
            df = pd.read_csv(f, sep="::", engine="python", header=None,
                             names=["userId", "movieId", "rating", "timestamp"])
    df.to_csv("ratings_1m.csv", index=False)
    print(f"saved ratings_1m.csv ({len(df):,} ratings)")


def load_ratings(path="ratings.csv"):
    if not os.path.exists(path):
        download_movielens()
    return pd.read_csv(path)


def load_movies(path="movies.csv"):
    if not os.path.exists(path):
        download_movielens()
    return pd.read_csv(path)


def temporal_split(df, test_frac=0.2, min_ratings=20):
    """Hold out each user's most recent `test_frac` of ratings."""
    df = df[df.groupby("userId")["movieId"].transform("size") >= min_ratings]
    df = df.sort_values(["userId", "timestamp"])

    train_parts, test_parts = [], []
    for _, grp in df.groupby("userId", sort=False):
        n_test = max(1, int(round(len(grp) * test_frac)))
        train_parts.append(grp.iloc[:-n_test])
        test_parts.append(grp.iloc[-n_test:])

    train = pd.concat(train_parts, ignore_index=True)
    test = pd.concat(test_parts, ignore_index=True)
    return train, test


def index_maps(train):
    """Map raw ids to contiguous integer indices for matrix models."""
    users = np.sort(train.userId.unique())
    items = np.sort(train.movieId.unique())
    return (
        {u: i for i, u in enumerate(users)},
        {m: i for i, m in enumerate(items)},
    )


if __name__ == "__main__":
    download_movielens()
    ratings = load_ratings()
    train, test = temporal_split(ratings)
    u_map, i_map = index_maps(train)
    print(f"train {len(train):,} | test {len(test):,}")
    print(f"users {train.userId.nunique()} | items in train {len(i_map)}")
    cold = set(test.movieId.unique()) - set(train.movieId.unique())
    print(f"test ratings on items never seen in training: "
          f"{test.movieId.isin(cold).sum():,} ({100*test.movieId.isin(cold).mean():.1f}%)")
