# Linear Regression from Scratch

## 📌 Problem Definition
Linear Regression is a supervised learning algorithm used to model the relationship between one or more input variables and a continuous target variable by fitting a linear equation.

---

## 📐 Mathematical Model

Hypothesis function:
\[
\hat{y} = Xw + b
\]

Where:
- \(X \in \mathbb{R}^{n \times d}\) is the input feature matrix  
- \(w \in \mathbb{R}^{d}\) is the weight vector  
- \(b \in \mathbb{R}\) is the bias term  

---

## 📉 Loss Function (Mean Squared Error)

\[
L = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
\]

The loss penalizes larger errors more heavily due to the squared term.

---

## ⚙️ Optimization
We minimize the loss using **Gradient Descent**, iteratively updating parameters in the direction of steepest descent.

---

## 🧠 Assumptions
- Linear relationship between input and output
- Independence of observations
- Homoscedasticity (constant variance)
- No strong multicollinearity

---

## 🚧 Limitations
- Sensitive to outliers
- Cannot model non-linear relationships without feature engineering
