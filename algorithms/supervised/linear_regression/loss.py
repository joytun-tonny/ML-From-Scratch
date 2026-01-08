import numpy as np

def mean_squared_error(y_true, y_pred):
    """
    Computes Mean Squared Error loss.
    """
    return np.mean((y_true - y_pred) ** 2)