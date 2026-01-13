# Logistic Regression from Scratch

## 📌 Problem Definition
Logistic Regression is a supervised learning algorithm used for **binary classification**.  
It models the probability that an input belongs to a particular class using a sigmoid function.

---

## 📐 Mathematical Model

Linear combination:
\[
z = Xw + b
\]

Sigmoid activation:
\[
\sigma(z) = \frac{1}{1 + e^{-z}}
\]

Predicted probability:
\[
\hat{y} = \sigma(Xw + b)
\]

---

## 📉 Loss Function (Binary Cross-Entropy)

\[
L = -\frac{1}{n} \sum [y \log(\hat{y}) + (1-y)\log(1-\hat{y})]
\]

This loss penalizes confident but incorrect predictions heavily.

---

## ⚙️ Optimization
We use **Gradient Descent** to minimize the loss by computing gradients with respect to:
- Weights \(w\)
- Bias \(b\)

---

## 🧠 Key Assumptions
- Linear decision boundary
- Independent features
- Binary target variable

---

## 🚧 Limitations
- Cannot model non-linear boundaries without feature engineering
- Sensitive to outliers
