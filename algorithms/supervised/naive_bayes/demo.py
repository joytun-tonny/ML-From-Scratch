import numpy as np
from model import GaussianNaiveBayes

np.random.seed(42)
X_class0 = np.random.randn(50, 2) + np.array([0, 0])
X_class1 = np.random.randn(50, 2) + np.array([3, 3])

X_train = np.vstack([X_class0, X_class1])
y_train = np.array([0] * 50 + [1] * 50)

model = GaussianNaiveBayes()
model.fit(X_train, y_train)

predictions = model.predict(X_train)
accuracy = np.mean(predictions == y_train)

print("Predictions (first 10):", predictions[:10])
print("Accuracy:", accuracy)
