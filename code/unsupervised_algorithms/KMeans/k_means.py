#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Implementation of K-Means algorithm."""

import multiprocessing
import sys

import numpy as np
import pandas as pd

from init_methods import k_means_plus_plus


class DistanceFromExamples:
    """Convenience class for easier multithreaded calculation of distance
    between clusters and data points."""

    def __init__(self, X):
        """Packs X data points for further use (less calls).

           Parameters
           ----------
           X : np.array or pd.DataFrame
               Dataset containing points to be clustered. Shape (n_examples,
               n_features)

        """
        self.X = X

    def __call__(self, c):
        """Returns distance between data points in X and cluster c.

           Parameters
           ----------
           c : np.array
               Vector containing features of given cluster.
               Shape (n_features, )

           Returns
           -------
           np.array
               Vector containing distances of each data point from cluster.
               Shape (n_examples, )
        """
        return np.linalg.norm(self.X - c, axis=1)


class KMeans:
    """KMeans (short summary)."""

    def __init__(self,
                 *,
                 n_clusters: int = 8,
                 initialization='k-means++',
                 tolerance: int = 1e-4,
                 max_iterations: int = 300,
                 n_jobs: int = None):
        """Setup for K-Means algorithm.

           Parameters
           ----------
           n_clusters : int, optional
               Number of clusters to create for KMeans (default is 8)
           initialization : str or array, optional
               Method of initialization, one of three strings should be
               provided:
                   a) 'k-means++': KMeans++ initialization algorithm,
                   see init_methods.py (default choice)
                   b) 'random': randomly initialize clusters
                   c) numpy array: array of shape (n_clusters, n_features)
                   containing centroids for algorithm
           tolerance : int, optional
               If distance between previous and current centroids is smaller
               than tolerance, finish training (default: 1e-4)
           max_iterations : int, optional
               Maximum number of iterations to perform during training (update
               clusters at most max_iterations times) (default: 300)
           n_jobs: int, optional
               Number of parallel jobs used for cluster calculations (default:
               number of cores).

               Each cluster may be computed in parallel, so maximum can be set
               by number of clusters (e.g. 3 clusters with n_jobs=8 would still
               give parallelism equal to 3)

        """
        self.n_clusters = n_clusters
        self.initialization = initialization
        self.max_iterations = max_iterations

        self.tolerance = tolerance
        if self.n_jobs = None:
            self.n_jobs = n_jobs if n_jobs <= n_clusters else n_clusters
        else:
            self.n_jobs - n_jobs
        self.labels = None
        self.cluster_centers = None

    def fit(self, X):
        """Trains classifier on given dataset.

           Algorithm:
               1. Create centroids using one of the methods from __init__.
               2. Assign each datapoint to the nearest cluster based on the
               distance :math: `D(x, c) = |c - x|`, where c is current cluster
               center and x is a data point.
               3. Calculate new centroids taking the mean of features from
               datapoints assigned to certain cluster.
               4. Repeat steps 2-3 until convergence or max_iterations is
               surpassed.

           Parameters
           ----------
           X : np.array or pd.DataFrame
               Dataset containing points to be clustered. Shape (n_examples,
               n_features)

           Throws
           -------
           ValueError
               ValueError with message 'Initialization array of wrong shape' if
               initialization parameter during object creation is np.array and
               is of wrong shape
           ValueError
               ValueError with message 'Unknown initialization scheme' if
               initialization is incorrect
        """
        # If X is pandas DataFrame transform it to numpy array
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

        # Create pool of processes to use for training
        with multiprocessing.Pool(processes=self.n_jobs) as pool:
            for _ in range(self.max_iterations):
                # Calculate distance from cluster for each data point
                distances_from_clusters = np.array(
                    pool.map(DistanceFromExamples(X), self.cluster_centers))

                # Assign each data point to cluster based on minimum distance
                examples_clustered = np.argmin(distances_from_clusters, axis=0)

                # Create mask with indices for each cluster, e.q. cluster one has
                # mask [1,0,0,1,0] which means elements [0,4] will be chosen from
                # the dataset for it.

                masks = np.array([
                    np.equal(examples_clustered, cluster)
                    for cluster in np.unique(examples_clustered)
                ])

                # Slice array with the masks and calculate mean for all
                # datapoints assigned to given cluster. Effectively creates new
                # cluster centroids
                new_cluster_centers = np.array(
                    [np.mean(X[mask], axis=0) for mask in masks])

                # Calculate distance between previous and current cluster
                iteration_distance = np.linalg.norm(
                    self.cluster_centers - new_cluster_centers)

                # If smaller than tolerance, eng algorithm
                if iteration_distance < self.tolerance:
                    return

                self.cluster_centers = new_cluster_centers

    def predict(self, X):
        """Predict clusters for given dataset.

           Chooses cluster based on minimum distance between data point and
           forementioned.

           Parameters
           ----------
           X : np.array or pd.DataFrame
               Dataset containing points to be clustered. Shape (n_examples,
               n_features)

           Returns
           -------
           np.array
               Array containing predicted clusters. Shape (n_examples,)
        """
        X = X.values if isinstance(X, pd.DataFrame) else X
        # Calculate disance between clusters and every data point
        distances_from_clusters = np.array(
            multiprocessing.Pool(processes=8).map(
                DistanceFromExamples(X), self.cluster_centers))

        # Choose the smallest distance
        return np.argmin(distances_from_clusters, axis=0)

    def fit_predict(self, X):
        """Convenience methods calling fit and predict afterwards.

           Parameters
           ----------
           X : np.array or pd.DataFrame
               Dataset containing points to be clustered. Shape (n_examples,
               n_features)

           Returns
           -------
           np.array
               Array containing predicted clusters. Shape (n_examples,)
        """
        self.fit(X)
        return self.predict(X)
