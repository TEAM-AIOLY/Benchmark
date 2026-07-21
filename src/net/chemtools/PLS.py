import torch

class PLS:
    def __init__(self, ncomp, weights=None):
        self.ncomp = ncomp
        self.weights = weights

    def fit(self, X, Y):
        # Convert input arrays to PyTorch tensors
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float64).clone().detach()
        if not isinstance(Y, torch.Tensor):
            Y = torch.tensor(Y, dtype=torch.float64).clone().detach()

        n, zp = X.shape
        q = Y.shape[1]

        if self.weights is None:
            self.weights = torch.ones(n, dtype=torch.float64) / n
        else:
            self.weights = torch.tensor(self.weights, dtype=torch.float64).clone().detach() / torch.sum(self.weights)

        self.xmeans = torch.sum(self.weights[:, None] * X, dim=0)
        X = X - self.xmeans

        self.ymeans = torch.sum(self.weights[:, None] * Y, dim=0)
        Y = Y - self.ymeans

        self.T = torch.zeros((n, self.ncomp), dtype=torch.float64)
        self.R = torch.zeros((zp, self.ncomp), dtype=torch.float64)
        self.W = torch.zeros((zp, self.ncomp), dtype=torch.float64)
        self.P = torch.zeros((zp, self.ncomp), dtype=torch.float64)
        self.C = torch.zeros((q, self.ncomp), dtype=torch.float64)
        self.TT = torch.zeros(self.ncomp, dtype=torch.float64)

        Xd = self.weights[:, None] * X
        tXY = Xd.T @ Y

        for a in range(self.ncomp):
            if q == 1:
                w = tXY[...,0]
            else:
                u, _, _ = torch.svd(tXY.T, some=False)
                u = u[:, 0]
                w = tXY @ u

            w = w / torch.sqrt(torch.sum(w * w))

            r = w.clone()
            if a > 0:
                for j in range(a):
                    r = r - torch.sum(self.P[:, j] * w) * self.R[:, j]

            t = X @ r
            tt = torch.sum(self.weights * t * t)

            c = (tXY.T @ r) / tt
            p = (Xd.T @ t) / tt

            tXY = tXY - (p[:, None] @ c[None]) * tt

            self.T[:, a] = t
            self.P[:, a] = p
            self.W[:, a] = w
            self.R[:, a] = r
            self.C[:, a] = c
            self.TT[a] = tt

    def transform(self, X):
        X = torch.tensor(X, dtype=torch.float64)
        X = X - self.xmeans
        T_new = X @ self.R
        return T_new

    def fit_transform(self, X, Y):
        self.fit(X, Y)
        return self.T

    def get_params(self):
        return {
            "T": self.T,
            "P": self.P,
            "W": self.W,
            "C": self.C,
            "R": self.R,
            "TT": self.TT,
            "xmeans": self.xmeans,
            "ymeans": self.ymeans,
            "weights": self.weights,
            "T.ortorcho": True
        }

    def predict(self, X, nlv=None):
        X = torch.tensor(X, dtype=torch.float64).clone().detach()
        X = X - self.xmeans

        if nlv is None:
            nlv = self.ncomp
        else:
            nlv = min(nlv, self.ncomp)

        B = self.W[:, :nlv] @ torch.inverse(self.P[:, :nlv].T @ self.W[:, :nlv]) @ self.C[:, :nlv].T
        predictions = X @ B + self.ymeans
        return predictions


def getknn(Xr, Xu, k, diss="euclidean"):
    n = Xr.shape[0]
    m = Xu.shape[0]

    if diss == "euclidean":
        distances = torch.cdist(Xu, Xr)
    elif diss == "mahalanobis":
        cov = torch.cov(Xr.T)
        cov_inv = torch.inverse(cov)
        diff = Xu.unsqueeze(1) - Xr.unsqueeze(0)
        distances = torch.sqrt(torch.sum(torch.matmul(diff, cov_inv) * diff, dim=2))
    elif diss == "correlation":
        Xr_mean = torch.mean(Xr, dim=0)
        Xu_mean = torch.mean(Xu, dim=0)
        Xr_centered = Xr - Xr_mean
        Xu_centered = Xu - Xu_mean
        Xr_norm = torch.norm(Xr_centered, dim=1)
        Xu_norm = torch.norm(Xu_centered, dim=1)
        distances = 1 - torch.matmul(Xu_centered, Xr_centered.T) / (Xu_norm.unsqueeze(1) * Xr_norm.unsqueeze(0))
    else:
        raise ValueError(f"Unknown distance type: {diss}")

    knn_indices = torch.argsort(distances, dim=1)[:, :k]
    knn_distances = torch.gather(distances, 1, knn_indices)

    return {"listnn": knn_indices, "listd": knn_distances}