#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Methods for KMeans initialization."""

import multiprocessing

import numpy as np


def k_means_plus_plus(X, n_clusters: int):
    """K-Means++ cluster initialization method.

       Computes clusters based on distance between data points.

       Algorithm:
           1. Choose random data point as initial cluster.
           2. Remove lastly chosen datapoint from dataset.
           3. Calculate distance between current cluster and all remaining data
           points (using :math: `|c - x|`, where c is cluster and x is data
           point).
           4. Assign probabilities to each cluster based on the distance
           :math: `p(x) = |c-x|^2` and create weighted probability dividing by
           sum of all distances.
           5. Randomly choose new cluster based on weighted probability array
           and append it to clusters list.
           6. Repeat steps 2-5 until all clusters are assigned

       Parameters
       ----------
       X : np.ndarray
           Dataset array containing features (points) from which clusters are
           chosen. Shape (n_examples, n_features)

       n_clusters: int
           Number of clusters to create

       Returns
       -------
       np.ndarray
           Matrix containg cluster points. Shape (n_clusters, n_features)
    """

    # Choose random row as initial centroid
    random_row = np.random.randint(0, X.shape[0])
    centroids = [X[random_row]]
    # Remove it from array
    cleared_array = np.delete(X, random_row, axis=0)

    # First cluster is assigned so range decreases by one
    for i in range(n_clusters - 1):
        # Calculate distance between cluster and all points
        squared_distances = np.linalg.norm(
            centroids[i] - cleared_array, axis=1)**2
        # Create weighted probability dividing by sym
        probabilities = squared_distances / np.sum(squared_distances)

        # Randomly select new centroid based on weighted probability array
        random_row = np.random.choice(len(probabilities), p=probabilities)

        # Add new centroid and remove it from the dataset
        centroids.append(X[random_row])
        cleared_array = np.delete(X, random_row, axis=0)

    return np.array(centroids)
