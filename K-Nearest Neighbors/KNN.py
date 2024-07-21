from collections import Counter
import numpy as np


def euclidean_distance(x1, x2):
    distance = np.sqrt(np.sum((x1 - x2) ** 2))
    return distance


class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return predictions

    def _predict(self, x):
        # compute the distances
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]

        # get closest k points
        k_closest_indices = np.argsort(distances)[: self.k]
        k_closest_labels = [self.y_train[i] for i in k_closest_indices]

        # majority vote
        most_common = Counter(k_closest_labels).most_common()

        return most_common[0][0]
