"""Three rating predictors, each exposing the same fit/predict interface.

All of them must handle users or items unseen in training, so every model falls
back through a chain: item mean -> user mean -> global mean.
"""
import numpy as np
import pandas as pd


class Baseline:
    """Global mean plus a user bias and an item bias.

    Deliberately simple. Much of the signal in ratings is not 'who likes what'
    but 'this user rates generously' and 'this film is well liked', and a model
    that cannot beat this is not learning anything about taste.
    """

    def fit(self, train):
        self.mu = train.rating.mean()
        self.item_bias = (train.groupby("movieId").rating.mean() - self.mu)
        self.user_bias = (train.groupby("userId").rating.mean() - self.mu)
        return self

    def predict(self, users, items):
        ub = self.user_bias.reindex(users).fillna(0.0).to_numpy()
        ib = self.item_bias.reindex(items).fillna(0.0).to_numpy()
        return np.clip(self.mu + ub + ib, 0.5, 5.0)


class UserCF:
    """User-based collaborative filtering with cosine similarity.

    Ratings are mean-centred per user before computing similarity, so that two
    users count as similar when they agree about what is above or below their
    own average, not when they happen to rate on the same part of the scale.
    """

    def __init__(self, k=40, shrinkage=10.0):
        self.k = k
        self.shrinkage = shrinkage

    def fit(self, train):
        self.mu = train.rating.mean()
        self.users = np.sort(train.userId.unique())
        self.items = np.sort(train.movieId.unique())
        self.u_idx = {u: i for i, u in enumerate(self.users)}
        self.i_idx = {m: i for i, m in enumerate(self.items)}

        R = np.zeros((len(self.users), len(self.items)), dtype=np.float32)
        mask = np.zeros_like(R, dtype=bool)
        ui = train.userId.map(self.u_idx).to_numpy()
        ii = train.movieId.map(self.i_idx).to_numpy()
        R[ui, ii] = train.rating.to_numpy()
        mask[ui, ii] = True

        self.user_mean = np.where(mask.sum(1) > 0,
                                  R.sum(1) / np.maximum(mask.sum(1), 1),
                                  self.mu)
        Rc = np.where(mask, R - self.user_mean[:, None], 0.0)

        norms = np.linalg.norm(Rc, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        Rn = Rc / norms
        sim = Rn @ Rn.T

        # Shrink similarity toward zero when two users share few items. Two
        # users overlapping on three films can look perfectly similar by luck.
        overlap = (mask.astype(np.float32) @ mask.T.astype(np.float32))
        sim *= overlap / (overlap + self.shrinkage)
        np.fill_diagonal(sim, 0.0)

        self.sim = sim
        self.R = R
        self.mask = mask
        self.item_mean = np.where(mask.sum(0) > 0,
                                  R.sum(0) / np.maximum(mask.sum(0), 1),
                                  self.mu)
        return self

    def _predict_one(self, u, i):
        if u not in self.u_idx:
            return self.item_mean[self.i_idx[i]] if i in self.i_idx else self.mu
        ui = self.u_idx[u]
        if i not in self.i_idx:
            return self.user_mean[ui]
        ii = self.i_idx[i]

        raters = np.where(self.mask[:, ii])[0]
        if len(raters) == 0:
            return self.user_mean[ui]

        s = self.sim[ui, raters]
        keep = np.argsort(-s)[: self.k]
        raters, s = raters[keep], s[keep]
        pos = s > 0
        if not pos.any():
            return self.user_mean[ui]
        raters, s = raters[pos], s[pos]

        devs = self.R[raters, ii] - self.user_mean[raters]
        pred = self.user_mean[ui] + (s @ devs) / s.sum()
        return pred

    def predict(self, users, items):
        out = np.fromiter(
            (self._predict_one(u, i) for u, i in zip(users, items)),
            dtype=np.float64, count=len(users),
        )
        return np.clip(out, 0.5, 5.0)


class MatrixFactorisation:
    """Biased matrix factorisation trained by stochastic gradient descent.

    Each user and each item gets a short vector of latent factors. Nobody tells
    the model what the factors mean; they are whatever best explains the ratings.
    """

    def __init__(self, n_factors=32, n_epochs=30, lr=0.008, reg=0.06, seed=42,
                 use_biases=True):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr = lr
        self.reg = reg
        self.seed = seed
        self.use_biases = use_biases

    def fit(self, train, verbose=False):
        rng = np.random.default_rng(self.seed)
        self.mu = train.rating.mean()
        self.users = np.sort(train.userId.unique())
        self.items = np.sort(train.movieId.unique())
        self.u_idx = {u: i for i, u in enumerate(self.users)}
        self.i_idx = {m: i for i, m in enumerate(self.items)}

        n_u, n_i = len(self.users), len(self.items)
        self.P = rng.normal(0, 0.05, (n_u, self.n_factors))
        self.Q = rng.normal(0, 0.05, (n_i, self.n_factors))
        self.bu = np.zeros(n_u)
        self.bi = np.zeros(n_i)

        ui = train.userId.map(self.u_idx).to_numpy()
        ii = train.movieId.map(self.i_idx).to_numpy()
        r = train.rating.to_numpy()

        order = np.arange(len(r))
        self.history = []
        for epoch in range(self.n_epochs):
            rng.shuffle(order)
            sq = 0.0
            for n in order:
                u, i, rui = ui[n], ii[n], r[n]
                pred = self.mu + self.bu[u] + self.bi[i] + self.P[u] @ self.Q[i]
                e = rui - pred
                sq += e * e

                if self.use_biases:
                    self.bu[u] += self.lr * (e - self.reg * self.bu[u])
                    self.bi[i] += self.lr * (e - self.reg * self.bi[i])
                pu, qi = self.P[u].copy(), self.Q[i]
                self.P[u] += self.lr * (e * qi - self.reg * pu)
                self.Q[i] += self.lr * (e * pu - self.reg * qi)

            rmse = np.sqrt(sq / len(r))
            self.history.append(rmse)
            if verbose:
                print(f"  epoch {epoch+1:2d}  train RMSE {rmse:.4f}")

        self.item_mean = train.groupby("movieId").rating.mean()
        self.user_mean = train.groupby("userId").rating.mean()
        return self

    def predict(self, users, items):
        out = np.empty(len(users))
        for n, (u, i) in enumerate(zip(users, items)):
            if u in self.u_idx and i in self.i_idx:
                a, b = self.u_idx[u], self.i_idx[i]
                out[n] = self.mu + self.bu[a] + self.bi[b] + self.P[a] @ self.Q[b]
            elif i in self.i_idx:
                out[n] = self.item_mean.get(i, self.mu)
            elif u in self.u_idx:
                out[n] = self.user_mean.get(u, self.mu)
            else:
                out[n] = self.mu
        return np.clip(out, 0.5, 5.0)


    def score_all(self, user):
        """Score every known item for one user at once.

        The per-item predict loop is fine for evaluation but far too slow
        inside the feedback simulation, where every user is scored against the
        whole catalogue in every round. This is the same arithmetic as a matrix
        product, so we do it that way.
        """
        if user not in self.u_idx:
            return np.full(len(self.items), self.mu), self.items
        a = self.u_idx[user]
        scores = self.mu + self.bu[a] + self.bi + self.Q @ self.P[a]
        return np.clip(scores, 0.5, 5.0), self.items
