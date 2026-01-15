import numpy as np
from distance import euclidean_distance


class KNNFromScratch:
    def __init__(self, k=3, task="classification"):
        self.k = k
        self.task = task

    def fit(self, X, y):
        """
        Store training data.
        """
        self.X_train = X
        self.y_train = y

    def _predict_single(self, x):
        # Compute distances to all training points
        distances = [
            euclidean_distance(x, x_train)
            for x_train in self.X_train
        ]

        # Get indices of k nearest neighbors
        k_indices = np.argsort(distances)[: self.k]
        k_nearest_labels = self.y_train[k_indices]

        # Classification or Regression
        if self.task == "classification":
            values, counts = np.unique(k_nearest_labels, return_counts=True)
            return values[np.argmax(counts)]
        else:
            return np.mean(k_nearest_labels)

    def predict(self, X):
        """
        Predict labels for all samples in X.
        """
        return np.array([self._predict_single(x) for x in X])
