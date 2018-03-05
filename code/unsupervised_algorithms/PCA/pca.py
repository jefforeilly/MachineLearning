import sys

import numpy as np
import pandas as pd


class PCA:
    def __init__(self, percentage: float = 0.95, solver: str = 'correlation'):
        self.percentage = percentage
        self.solver = solver
        if solver == 'correlation':
            self.solver = lambda data: np.linalg.eig(np.corrcoef(data))
        elif solver == 'covariance':
            self.solver = lambda data: np.linalg.eig(np.cov(data))
        elif solver == 'svd':
            self.solver = lambda data: np.linalg.svd(data)[:2]
        else:
            print(
                'Unknown argument for solver. Either \'correlation\''
                'or \'covariance\' allowed',
                file=sys.stderr)

        self.explained_variance = None
        self.eigenvectors = None
        self.eigenvalues = None
        self.projection_matrix = None

    def fit(self, X):
        data = X.values if isinstance(X, pd.DataFrame) else X
        self.eigenvectors, self.eigenvalues = self.solver(data.T)
        self.explained_variance = np.cumsum(self.eigenvalues) / sum(
            self.eigenvalues)

        satisfying_index = np.argwhere(
            self.explained_variance >= self.percentage)[-1]
        self.projection_matrix = self.eigenvectors[:satisfying_index]

    def transform(self, X):
        data = X.values if isinstance(X, pd.DataFrame) else X
        return np.dot(data, self.projection_matrix)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, inv_X):
        pass
