import torch


class Ridge:
    def __init__(self, alpha=0, rank=None, fit_intercept=True):
        self.alpha = alpha
        self.rank = rank
        self.fit_intercept = fit_intercept

    def fit(self, X, Y):
        if self.fit_intercept:
            X = torch.cat([torch.ones(X.shape[0], 1), X], dim = 1)
        
        ridge = self.alpha*torch.eye(X.shape[0])
        # Ridge formulation : (X.T @ X + lambda * I)^{-1} @ B = X.T @ Y

        normal_matrix = X.T @ X
        moment_matrix = X.T @ Y

        ridge = self.alpha*torch.eye(normal_matrix.shape[0])

        self.W = torch.linalg.lstsq(normal_matrix + ridge, moment_matrix).solution

        if self.rank is not None:
            U, S, Vh = torch.svd(X @ self.W)
            Vhr = Vh[:, :self.rank]
            self.W = self.W @ Vhr @ Vhr.T

    def predict(self, X):
        if self.fit_intercept:
            X = torch.cat([torch.ones(X.shape[0], 1), X], dim = 1)
        
        return X @ self.W