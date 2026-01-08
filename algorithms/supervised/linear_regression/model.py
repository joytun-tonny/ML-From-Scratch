import numpy as np
from loss import mean_squared_error
from optimizer import GradientDescent


class LinearRegressionScratch:
    def __init__(self, lr=0.01, epochs=1000):
        self.lr = lr
        self.epochs = epochs
        self.loss_history = []

    def fit(self, X, y):
        """
        Train Linear Regression using Gradient Descent.
        """
        n_samples, n_features = X.shape

        # Initialize parameters
        self.w = np.zeros(n_features)
        self.b = 0.0

        optimizer = GradientDescent(self.lr)

        for _ in range(self.epochs):
            # Forward pass
            y_pred = np.dot(X, self.w) + self.b

            # Compute loss
            loss = mean_squared_error(y, y_pred)
            self.loss_history.append(loss)

            # Compute gradients
            dw = (-2 / n_samples) * np.dot(X.T, (y - y_pred))
            db = (-2 / n_samples) * np.sum(y - y_pred)

            # Update parameters
            self.w, self.b = optimizer.step(self.w, self.b, dw, db)

    def predict(self, X):
        """
        Predict continuous values for input X.
        """
        return np.dot(X, self.w) + self.b
