"""Experiment 2 - where does each model actually win, and what does it push?

Average error is a blunt summary. Here we ask three sharper questions:
  1. Does better rating accuracy translate into better recommendation lists?
  2. Do the models win everywhere, or only where data is dense?
  3. How heavily does each model lean on already-popular items?
"""
import json

import numpy as np
import pandas as pd

from data import load_ratings, temporal_split
from models import Baseline, UserCF, MatrixFactorisation
from evaluate import (rmse, mae, ranking_metrics, error_breakdown,
                      catalogue_coverage)


def build_models():
    return {
        "Baseline": Baseline(),
        "UserCF": UserCF(k=40, shrinkage=10.0),
        "MatrixFactorisation": MatrixFactorisation(n_factors=32, n_epochs=30),
    }


def catalogue_popularity_baseline(train):
    """Average popularity percentile of a randomly chosen catalogue item.

    This is the reference point. A model with no popularity bias would
    recommend items averaging around this value.
    """
    pop = train.groupby("movieId").size()
    return float(pop.rank(pct=True).mean())


def run(n=10):
    ratings = load_ratings()
    train, test = temporal_split(ratings)

    reference = catalogue_popularity_baseline(train)
    print(f"reference: mean popularity percentile of the catalogue = "
          f"{reference:.3f}\n")

    rows, breakdowns = [], {}
    for name, model in build_models().items():
        print(f"--- {name}")
        model.fit(train)

        pred = model.predict(test.userId.to_numpy(), test.movieId.to_numpy())
        acc = {"rmse": round(rmse(pred, test.rating.to_numpy()), 4),
               "mae": round(mae(pred, test.rating.to_numpy()), 4)}

        rank = ranking_metrics(model, train, test, n=n)
        cov = catalogue_coverage(model, train, n=n)

        row = {"model": name, **acc,
               **{k: (round(v, 4) if isinstance(v, float) else v)
                  for k, v in rank.items()},
               "catalogue_coverage": round(cov, 4)}
        rows.append(row)

        breakdowns[name] = error_breakdown(model, train, test)

        print(f"  RMSE {acc['rmse']:.4f}")
        print(f"  precision@{n} {rank[f'precision@{n}']:.4f}   "
              f"recall@{n} {rank[f'recall@{n}']:.4f}   "
              f"ndcg@{n} {rank[f'ndcg@{n}']:.4f}")
        print(f"  mean popularity of recommendations "
              f"{rank['mean_popularity_of_recs']:.3f} "
              f"(catalogue reference {reference:.3f})")
        print(f"  catalogue coverage {cov:.1%}")
        print()

    df = pd.DataFrame(rows)
    df.to_csv("results_ranking.csv", index=False)

    print("=== RMSE by user history and item popularity ===")
    for name, bd in breakdowns.items():
        print(f"\n{name}")
        for dim, vals in bd.items():
            print(f"  {dim}:")
            for bucket, v in vals.items():
                print(f"    {bucket:>22s}  RMSE {v}")

    with open("results_breakdown.json", "w") as f:
        json.dump({"catalogue_reference_popularity": reference,
                   "breakdowns": breakdowns}, f, indent=2)

    return df, breakdowns


if __name__ == "__main__":
    run()
