import multiprocessing
import sys

import numpy as np
import pandas as pd

from init_methods import k_means_plus_plus


class DistanceFromExamples:
    def __init__(self, X):
        self.X = X

    def __call__(self, a):
        return np.linalg.norm(self.X - a, axis=1)


class KMeans:
    def __init__(self,
                 *,
                 n_clusters: int = 8,
                 initialization='k-means++',
                 tolerance: int = 1e-4,
                 max_iterations: int = 300,
                 n_jobs: int = None):
        self.n_clusters = n_clusters
        self.initialization = initialization
        self.max_iterations = max_iterations

        self.tolerance = tolerance
        self.n_jobs = n_jobs
        self.labels = None
        self.cluster_centers = None

    def fit(self, X):
        X = X.values if isinstance(X, pd.DataFrame) else X

        # PARSE INITIALIZATION METHOD
        if self.initialization == 'k-means++':
            self.cluster_centers = k_means_plus_plus(X, self.n_clusters)
        elif self.initialization == 'random':
            self.cluster_centers = X[np.random.randint(
                X.shape[0], size=self.n_clusters)]
        elif isinstance(self.initialization, np.ndarray):
            if self.initialization.shape() != (self.n_clusters, X.shape[1]):
                raise ValueError('Initialization array of wrong shape')
        else:
            raise ValueError('Unknown initialization scheme')

        with multiprocessing.Pool(processes=self.n_jobs) as pool:
            for _ in range(self.max_iterations):
                distances_from_clusters = np.array(
                    pool.map(DistanceFromExamples(X), self.cluster_centers))

                examples_clustered = np.argmin(distances_from_clusters, axis=0)
                masks = np.array([
                    np.equal(examples_clustered, cluster)
                    for cluster in np.unique(examples_clustered)
                ])
                new_cluster_centers = np.array(
                    [np.mean(X[mask], axis=0) for mask in masks])

                iteration_distance = np.linalg.norm(
                    self.cluster_centers - new_cluster_centers)

                if iteration_distance < self.tolerance:
                    return

                self.cluster_centers = new_cluster_centers

    def predict(self, X):
        distances_from_clusters = np.array(
            multiprocessing.Pool(processes=8).map(
                DistanceFromExamples(X), self.cluster_centers))

        return np.argmin(distances_from_clusters, axis=0)

    def fit_predict(self, X):
        self.fit(X)
        return self.predict(X)
