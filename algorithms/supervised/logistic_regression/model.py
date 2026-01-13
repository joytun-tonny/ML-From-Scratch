import numpy as np
from loss import binary_cross_entropy
from optimizer import GradientDescent


class LogisticRegressionScratch:
    def __init__(self, lr=0.01, epochs=1000):
        self.lr = lr
        self.epochs = epochs

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        """
        Train Logistic Regression using Gradient Descent
        """
        n_samples, n_features = X.shape

        self.w = np.zeros(n_features)
        self.b = 0.0

        optimizer = GradientDescent(self.lr)

        for _ in range(self.epochs):
            # Forward pass
            linear_model = np.dot(X, self.w) + self.b
            y_pred = self._sigmoid(linear_model)

            # Gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            # Parameter update
            self.w, self.b = optimizer.step(self.w, self.b, dw, db)

    def predict_proba(self, X):
        linear_model = np.dot(X, self.w) + self.b
        return self._sigmoid(linear_model)

    def predict(self, X, threshold=0.5):
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)
