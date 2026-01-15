import numpy as np
from model import KNNFromScratch

# Simple dataset
X_train = np.array([
    [1, 2],
    [2, 3],
    [3, 3],
    [6, 5],
    [7, 7],
    [8, 6]
])

y_train = np.array([0, 0, 0, 1, 1, 1])

X_test = np.array([
    [2, 2],
    [7, 6]
])

# Train KNN
knn = KNNFromScratch(k=3, task="classification")
knn.fit(X_train, y_train)

# Predict
predictions = knn.predict(X_test)

print("Predictions:", predictions)
