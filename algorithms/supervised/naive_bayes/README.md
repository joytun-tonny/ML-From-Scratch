# Naive Bayes from Scratch

## Problem Definition
Naive Bayes is a probabilistic classifier based on **Bayes' Theorem** with the "naive" assumption of conditional independence between features.

---

## Mathematical Model

Bayes' Theorem:
\[
P(y|X) = \frac{P(X|y) \cdot P(y)}{P(X)}
\]

With the naive independence assumption:
\[
P(X|y) = \prod_{i=1}^{d} P(x_i|y)
\]

For Gaussian Naive Bayes, each feature likelihood is modeled as:
\[
P(x_i|y) = \frac{1}{\sqrt{2\pi\sigma_y^2}} \exp\left(-\frac{(x_i - \mu_y)^2}{2\sigma_y^2}\right)
\]

---

## Classification Rule

\[
\hat{y} = \arg\max_y \left[ \log P(y) + \sum_{i=1}^{d} \log P(x_i|y) \right]
\]

We use log-probabilities to avoid numerical underflow.

---

## Key Characteristics
- Very fast training and prediction
- Works well with high-dimensional data
- No iterative optimization required

---

## Limitations
- Strong independence assumption rarely holds in practice
- Sensitive to feature distributions that deviate from Gaussian
- Cannot capture feature interactions
