import numpy as np
from model import LogisticRegressionScratch

# Simple binary classification dataset
X = np.array([
    [1], [2], [3], [4], [5], [6]
])
y = np.array([0, 0, 0, 1, 1, 1])

# Train model
model = LogisticRegressionScratch(lr=0.1, epochs=2000)
model.fit(X, y)

# Predictions
predictions = model.predict(X)
probabilities = model.predict_proba(X)

print("Predicted probabilities:", probabilities)
print("Predicted classes:", predictions)
