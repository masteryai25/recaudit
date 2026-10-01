"""Experiment 1 - does the evaluation protocol change the conclusions?

Most published MovieLens results use a random split, which lets a model see a
user's later behaviour while predicting their earlier behaviour. A real system
never has that. We run every model under both protocols on the same data.
"""
import json
import time

import numpy as np
import pandas as pd

from data import load_ratings, temporal_split
from models import Baseline, UserCF, MatrixFactorisation
from evaluate import rmse, mae


def random_split(df, test_frac=0.2, min_ratings=20, seed=42):
    """Hold out a random 20% of each user's ratings, ignoring time."""
    rng = np.random.default_rng(seed)
    df = df[df.groupby("userId")["movieId"].transform("size") >= min_ratings]
    train_parts, test_parts = [], []
    for _, grp in df.groupby("userId", sort=False):
        idx = rng.permutation(len(grp))
        n_test = max(1, int(round(len(grp) * test_frac)))
        test_parts.append(grp.iloc[idx[:n_test]])
        train_parts.append(grp.iloc[idx[n_test:]])
    return (pd.concat(train_parts, ignore_index=True),
            pd.concat(test_parts, ignore_index=True))


def build_models():
    return {
        "Baseline": Baseline(),
        "UserCF": UserCF(k=40, shrinkage=10.0),
        "MatrixFactorisation": MatrixFactorisation(n_factors=32, n_epochs=30),
    }


def run():
    ratings = load_ratings()
    splits = {
        "random": random_split(ratings),
        "temporal": temporal_split(ratings),
    }

    results = []
    for split_name, (train, test) in splits.items():
        print(f"\n=== {split_name} split: "
              f"train {len(train):,} / test {len(test):,}")
        u = test.userId.to_numpy()
        i = test.movieId.to_numpy()
        a = test.rating.to_numpy()

        for name, model in build_models().items():
            t0 = time.time()
            model.fit(train)
            pred = model.predict(u, i)
            row = {
                "split": split_name,
                "model": name,
                "rmse": round(rmse(pred, a), 4),
                "mae": round(mae(pred, a), 4),
                "fit_seconds": round(time.time() - t0, 1),
            }
            results.append(row)
            print(f"  {name:22s} RMSE {row['rmse']:.4f}  "
                  f"MAE {row['mae']:.4f}  ({row['fit_seconds']}s)")

    df = pd.DataFrame(results)
    df.to_csv("results_split_comparison.csv", index=False)

    print("\n=== effect of the evaluation protocol ===")
    pivot = df.pivot(index="model", columns="split", values="rmse")
    pivot["difference"] = (pivot["temporal"] - pivot["random"]).round(4)
    pivot["percent_worse"] = (100 * pivot["difference"] / pivot["random"]).round(1)
    print(pivot.to_string())
    pivot.to_csv("results_split_effect.csv")

    with open("results_split_comparison.json", "w") as f:
        json.dump(results, f, indent=2)
    return df, pivot


if __name__ == "__main__":
    run()
