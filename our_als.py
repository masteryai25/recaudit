"""Implicit-feedback ALS, implemented from the paper, plus one modification.

Baseline method: Hu, Koren and Volinsky (2008), "Collaborative Filtering for
Implicit Feedback Datasets". Every user-item pair gets a target of 1 if the
interaction happened and 0 if it did not, and a confidence weight c = 1 + alpha*r
saying how much to trust that target. Unobserved pairs are not treated as
dislikes, only as weak evidence, which is what makes it work on sparse data.

Our modification. In the original, every unobserved pair carries the same
confidence of 1. That embeds an assumption worth questioning: not seeing a film
that almost nobody has seen is treated as just as informative as not seeing a
film that everybody has seen. The second is real evidence of disinterest; the
first is mostly evidence that it was never on offer. So we scale the confidence
of unobserved pairs by item popularity:

    c0_i = 1 + beta * popularity_percentile(i)

With beta = 0 this reduces exactly to the original method, which gives us a
control. With beta > 0, obscure items are penalised less for being unseen, so
they should compete more fairly for a place in the recommendation list.
"""
import numpy as np
import scipy.sparse as sp


class ALS:
    """Implicit ALS with optional popularity-aware confidence on unobserved pairs.

    Parameters
    ----------
    factors        : latent dimension
    regularization : L2 penalty
    iterations     : number of alternating sweeps
    alpha          : confidence scaling for observed interactions
    beta           : our modification. 0 reproduces Hu et al. exactly.
    """

    def __init__(self, factors=64, regularization=0.05, iterations=20,
                 alpha=40.0, beta=0.0, seed=42):
        self.factors = factors
        self.regularization = regularization
        self.iterations = iterations
        self.alpha = alpha
        self.beta = beta
        self.seed = seed

    def fit(self, mat, verbose=False):
        rng = np.random.default_rng(self.seed)
        self.mat = mat.tocsr()
        n_users, n_items = mat.shape

        self.U = rng.normal(0, 0.01, (n_users, self.factors))
        self.V = rng.normal(0, 0.01, (n_items, self.factors))

        # Popularity percentile per item, used only by the modification.
        pop = np.asarray(mat.sum(axis=0)).ravel()
        order = pop.argsort().argsort()
        pop_pct = order / max(len(order) - 1, 1)
        # w_i is the confidence attached to an unobserved pair for item i.
        self.w = 1.0 + self.beta * pop_pct

        matT = self.mat.T.tocsr()

        for it in range(self.iterations):
            self.U = self._solve(self.mat, self.V, self.w, weighted_by_item=True)
            self.V = self._solve(matT, self.U, None, weighted_by_item=False)
            if verbose:
                print(f"  iteration {it+1}/{self.iterations}")
        return self

    def _solve(self, R, Y, w, weighted_by_item):
        """Solve for one factor matrix holding the other fixed.

        The standard trick: precompute Y^T C0 Y once for all rows, then for each
        row add only the corrections for its observed entries, which are few.
        """
        n_rows = R.shape[0]
        f = self.factors
        reg = self.regularization * np.eye(f)

        if weighted_by_item and w is not None:
            YtCY = (Y * w[:, None]).T @ Y
        else:
            YtCY = Y.T @ Y

        X = np.zeros((n_rows, f))
        for row in range(n_rows):
            start, end = R.indptr[row], R.indptr[row + 1]
            idx = R.indices[start:end]
            if len(idx) == 0:
                continue
            vals = R.data[start:end]
            c = 1.0 + self.alpha * vals           # confidence for observed pairs
            base = w[idx] if (weighted_by_item and w is not None) else 1.0

            Yi = Y[idx]
            # (c - base) is the extra confidence beyond the unobserved default
            A = YtCY + (Yi * (c - base)[:, None]).T @ Yi + reg
            b = (Yi * c[:, None]).sum(axis=0)
            X[row] = np.linalg.solve(A, b)
        return X

    def recommend(self, user, n=10, filter_seen=True):
        scores = self.V @ self.U[user]
        if filter_seen:
            scores[self.mat[user].indices] = -np.inf
        return np.argpartition(-scores, n)[:n][
            np.argsort(-scores[np.argpartition(-scores, n)[:n]])]
