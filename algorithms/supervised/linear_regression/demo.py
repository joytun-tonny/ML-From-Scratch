import numpy as np
import matplotlib.pyplot as plt
from model import LinearRegressionScratch

# Sample dataset
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])

# Train model
model = LinearRegressionScratch(lr=0.01, epochs=2000)
model.fit(X, y)

# Predictions
y_pred = model.predict(X)

print("Predicted values:", y_pred)
print("Learned weight:", model.w)
print("Learned bias:", model.b)

# Visualization
plt.scatter(X, y, label="Actual data")
plt.plot(X, y_pred, color="red", label="Regression line")
plt.legend()
plt.title("Linear Regression From Scratch")
plt.show()
