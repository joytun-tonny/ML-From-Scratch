import numpy as np
import matplotlib.pyplot as plt
from model import DecisionTreeClassifier

np.random.seed(42)
X_class0 = np.random.randn(50, 2) + np.array([0, 0])
X_class1 = np.random.randn(50, 2) + np.array([3, 3])
X = np.vstack([X_class0, X_class1])
y = np.array([0] * 50 + [1] * 50)

tree = DecisionTreeClassifier(max_depth=5, min_samples_split=2)
tree.fit(X, y)
predictions = tree.predict(X)
acc = np.mean(predictions == y)

print("Accuracy:", acc)

# Decision boundary
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
grid = np.c_[xx.ravel(), yy.ravel()]
Z = tree.predict(grid).reshape(xx.shape)

plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap="RdBu")
plt.scatter(X[y == 0, 0], X[y == 0, 1], c="blue", s=20, label="Class 0")
plt.scatter(X[y == 1, 0], X[y == 1, 1], c="red", s=20, label="Class 1")
plt.title(f"Decision Tree Boundary (depth=5, Acc: {acc:.2f})")
plt.legend()
plt.show()
