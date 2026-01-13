import numpy as np

def binary_cross_entropy(y_true, y_pred):
    """
    Computes Binary Cross Entropy loss.
    """
    epsilon = 1e-15  # to avoid log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(
        y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)
    )
