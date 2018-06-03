import multiprocessing

import numpy as np
from sklearn.cluster import KMeans

class RIM:
    def __init__(self,
                 alpha: float,
                 n_clusters: int = 8,
                 n_init: int = 10,
                 max_iter: int = 300,
                 n_jobs: int = multiprocessing.cpu_count()):

        self.alpha = alpha
        self.n_jobs = n_jobs
        self.kmeans = KMeans(
            n_clusters,
            n_init,
            max_iter,
            n_jobs=n_jobs,
        )

    def fit(self, X, y):
        train = self.normalize(X)
        self.kmeans.fit(train)
        # NA PONIŻSZYM FITUJEMY REGRESJĘ LOGISTYCZNĄ
        self.labels = self.kmeans.labels_

    def normalize(self, X):
        return (X - np.mean(X, axis=1)) / np.std(X, axis=1, ddof=1)
