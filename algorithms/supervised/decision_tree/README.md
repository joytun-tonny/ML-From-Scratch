# Decision Tree from Scratch

## Problem Definition
A Decision Tree is a supervised learning algorithm that splits data into branches based on feature thresholds, creating a tree structure for classification or regression.

---

## Mathematical Model

### Entropy
\[
H(S) = -\sum_{i=1}^{c} p_i \log_2(p_i)
\]

Where \(p_i\) is the proportion of class \(i\) in set \(S\).

### Information Gain
\[
IG(S, A) = H(S) - \sum_{v \in \text{values}(A)} \frac{|S_v|}{|S|} H(S_v)
\]

The algorithm selects the feature and threshold that maximizes information gain at each split.

---

## Algorithm Steps
1. Compute entropy of the current node
2. For each feature and threshold, compute information gain
3. Split on the feature/threshold with the highest gain
4. Recursively build left and right subtrees
5. Stop when max depth is reached or node is pure

---

## Key Characteristics
- Non-parametric: no assumptions about data distribution
- Interpretable: easy to visualize and explain
- Handles both numerical and categorical data

---

## Limitations
- Prone to overfitting without pruning or depth limits
- Unstable: small changes in data can produce different trees
- Biased toward features with more levels
