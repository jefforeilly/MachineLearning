import sys

import numpy as np
import pandas as pd

import optimizers


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))


class LassoRegularization:
    def __init__(self, alpha):
        self.alpha = alpha

    def cost(self, theta):
        return self.alpha * np.sum(np.abs(theta))

    def gradient(self, theta):
        return 2 * self.alpha * np.abs(theta) / theta


class TikhonovRegularization:
    def __init__(self, alpha):
        self.alpha = alpha

    def cost(self, theta):
        return self.alpha * np.sum(theta**2)

    def gradient(self, theta):
        return 2 * self.alpha * theta


class CostFunction:
    def __init__(self, X, y, alpha, regularization):
        self.X = X.T
        self.y = y
        self.alpha = alpha
        self.regularization = regularization
        self.unique_labels = np.unique(y)

    def gradient(self, weights):
        predictions = np.dot(weights, self.X)

        probability_predictions = np.array(
            [softmax(prediction) for prediction in predictions])
        best_hypothesis = np.argmax(probability_predictions, axis=0)
        best_weights = weights[best_hypothesis].T
        # print(accuracy_score(self.y, best_hypothesis))

        class_predictions = np.array(
            np.ma.make_mask([
                best_hypothesis == current for current in self.unique_labels
            ]))

        y_true = np.array(
            np.ma.make_mask(
                [self.y == current for current in self.unique_labels]))

        probability_predictions = np.sum(best_weights * self.X, axis=0)

        # ADD REGULARIZATION
        gradient = np.mean(
            np.array(
                [((class_prediction == current_y_true) - softmax(prediction)) *
                 self.X
                 for class_prediction, current_y_true, prediction in zip(
                     class_predictions, y_true, predictions)]),
            axis=2)
        gradient += np.array(
            [self.regularization.gradient(weight) for weight in weights])

        return (0., gradient)


class MaxEnt:
    def __init__(self,
                 *,
                 regularization: str = 'l2',
                 optimizer: str = 'adagrad',
                 alpha: float = 0.1,
                 max_iter: int = 200):

        # PARSE REGULARIZATION CHOICE
        if regularization == 'l1':
            self.regularization = LassoRegularization(alpha)
        elif regularization == 'l2':
            self.regularization = TikhonovRegularization(alpha)
        else:
            print('Unknown regularization method, \
                see documentation for details.')

        # PARSE OPTIMIZER CHOICE
        if optimizer == 'gradient_descent':
            self.optimizer = optimizers.gradient_descent
        elif optimizer == 'momentum':
            self.optimizer = optimizers.momentum
        elif optimizer == 'adagrad':
            self.optimizer = optimizers.adagrad
        elif optimizer == 'nesterov':
            self.optimizer = optimizers.nesterov
        else:
            self.optimizer = optimizers.gradient_descent
            print(
                'Incorrect choice of optimizer, see documentation.',
                file=sys.stderr)

        self.alpha = alpha
        self.weights = None
        self.max_iter = max_iter

    def fit(self, X, y):
        X = X.values if isinstance(X, pd.DataFrame) else X
        y = y.values if isinstance(X, pd.DataFrame) else y

        unique_values, _ = np.unique(y, return_counts=True)

        class_indices = np.array(
            np.ma.make_mask([y == current for current in unique_values]))

        class_datasets = np.array([X[indices] for indices in class_indices])
        classes_metrics = np.array(
            [dataset.sum(axis=0) for dataset in class_datasets]).reshape(
                len(unique_values), -1)
        self.weights = classes_metrics / \
            classes_metrics.sum(axis=1)[:, np.newaxis]

        optimizer_iterator = iter(
            self.optimizer(
                CostFunction(X, y, self.alpha, self.regularization),
                self.weights,
            ))

        for _ in range(self.max_iter):
            self.weights, _, _ = next(optimizer_iterator)

    def predict_proba(self, X):
        predictions = np.dot(self.weights, X.T)
        return np.array([softmax(prediction) for prediction in predictions])

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=0)
