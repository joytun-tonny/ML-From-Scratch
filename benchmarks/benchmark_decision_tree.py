import numpy as np
import time
import sys
sys.path.append("../algorithms/supervised/decision_tree")
from model import DecisionTreeClassifier as ScratchTree

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

np.random.seed(42)
X_class0 = np.random.randn(100, 2) + np.array([0, 0])
X_class1 = np.random.randn(100, 2) + np.array([3, 3])
X = np.vstack([X_class0, X_class1])
y = np.array([0] * 100 + [1] * 100)

print("=" * 60)
print("BENCHMARK: Decision Tree")
print("=" * 60)

# Scratch
start = time.time()
scratch_model = ScratchTree(max_depth=5)
scratch_model.fit(X, y)
scratch_pred = scratch_model.predict(X)
scratch_time = time.time() - start
scratch_acc = accuracy_score(y, scratch_pred)

# Scikit-learn
start = time.time()
sklearn_model = DecisionTreeClassifier(max_depth=5)
sklearn_model.fit(X, y)
sklearn_pred = sklearn_model.predict(X)
sklearn_time = time.time() - start
sklearn_acc = accuracy_score(y, sklearn_pred)

print(f"\n{'Metric':<25} {'Scratch':>15} {'Scikit-Learn':>15}")
print("-" * 55)
print(f"{'Accuracy':<25} {scratch_acc:>15.4f} {sklearn_acc:>15.4f}")
print(f"{'Training Time (s)':<25} {scratch_time:>15.6f} {sklearn_time:>15.6f}")
