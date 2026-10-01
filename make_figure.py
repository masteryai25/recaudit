"""The headline figure.

One chart: accuracy on one axis, personalisation on the other, every model
plotted. The point is that the two are not on a trade-off curve. If they were,
the models would fall along a line. They do not, and one model sits alone in the
region where a system looks accurate while recommending the same items to
everyone.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Results collected across the experiments. Shared items are out of 10, so the
# popularity reference of 4.65 is the line above which a model differentiates
# between users less than a system that ignores them entirely.
POP_REFERENCE = 4.65

MODELS = [
    # label,                        precision, shared_of_10, coverage, marker
    ("Baseline\n(biases only)",         0.0007, 0.34, 0.089, "o"),
    ("UserCF",                          0.0010, 0.55, 0.136, "o"),
    ("MF\n(rating-optimised)",          0.0150, 5.72, 0.023, "X"),
    ("MF minus biases",                 0.0410, 1.07, 0.062, "s"),
    ("ALS\n(ranking-optimised)",        0.0573, 0.35, 0.099, "D"),
]


def main():
    fig, ax = plt.subplots(figsize=(9, 6.2))

    # region where a model is less differentiated than popularity ranking
    ax.axhspan(POP_REFERENCE, 10, color="#d62728", alpha=0.07, zorder=0)
    ax.axhline(POP_REFERENCE, color="#d62728", ls="--", lw=1.4, zorder=1)
    ax.text(0.0605, POP_REFERENCE + 0.28,
            "less differentiated than ranking by popularity alone",
            color="#b22222", fontsize=9.5, ha="right", style="italic")

    for label, prec, shared, cov, marker in MODELS:
        colour = "#d62728" if shared >= POP_REFERENCE else "#2c7fb8"
        ax.scatter(prec, shared, s=120 + 5200 * cov, marker=marker,
                   color=colour, alpha=0.75, edgecolor="black",
                   linewidth=1.1, zorder=3)
        offsets = {"Baseline\n(biases only)": (-6, -46),
                   "UserCF": (14, 26),
                   "MF\n(rating-optimised)": (0, -40),
                   "MF minus biases": (0, 22),
                   "ALS\n(ranking-optimised)": (0, 22)}
        ax.annotate(label, (prec, shared), textcoords="offset points",
                    xytext=offsets.get(label, (0, 18)), ha="center",
                    fontsize=9.5, zorder=4)

    ax.set_xlabel("Accuracy  (precision@10)", fontsize=11.5)
    ax.set_ylabel("Users' lists overlap  (shared items out of 10)",
                  fontsize=11.5)
    ax.set_title("Accuracy does not predict whether a recommender personalises",
                 fontsize=13.5, pad=14)
    ax.set_ylim(-1.1, 6.9)
    ax.set_xlim(-0.004, 0.066)
    ax.grid(alpha=0.25, zorder=0)

    # marker size legend
    for cov, x in [(0.02, 0.046), (0.10, 0.054)]:
        ax.scatter(x, 3.55, s=120 + 5200 * cov, color="grey",
                   alpha=0.45, edgecolor="black", linewidth=0.8)
        ax.text(x, 3.05, f"{cov:.0%}", ha="center", fontsize=8.5, color="grey")
    ax.text(0.050, 4.05, "marker size = catalogue coverage",
            ha="center", fontsize=9, color="grey")

    fig.text(0.5, 0.012,
             "MovieLens 100K, temporal split. Lower on the vertical axis is "
             "more personalised. The rating-optimised model is the only one "
             "in the red region,\nand removing its bias terms moves it out "
             "while also tripling its accuracy.",
             ha="center", fontsize=8.8, color="#444444")

    plt.tight_layout(rect=[0, 0.055, 1, 1])
    plt.savefig("figure_accuracy_vs_personalisation.png", dpi=200)
    print("saved figure_accuracy_vs_personalisation.png")


if __name__ == "__main__":
    main()
