import multiprocessing

import numpy as np


def k_means_plus_plus(X, n_clusters: int):

    random_row = np.random.randint(0, X.shape[0])
    centroids = [X[random_row]]
    cleared_array = np.delete(X, random_row, axis=0)

    for i in range(n_clusters - 1):
        squared_distances = np.linalg.norm(
            centroids[i] - cleared_array, axis=1)**2
        probabilities = squared_distances / np.sum(squared_distances)

        random_row = np.random.choice(len(probabilities), p=probabilities)

        centroids.append(X[random_row])
        cleared_array = np.delete(X, random_row, axis=0)
    return np.array(centroids)
