# k-Nearest Neighbors (k-NN) from Scratch

## 📌 Problem Definition
k-Nearest Neighbors (k-NN) is a **non-parametric, instance-based learning algorithm** used for both classification and regression.  
It predicts the output of a sample based on the majority vote (classification) or average (regression) of its *k* nearest neighbors.

---

## 📐 Distance Metric
The most common distance metric is **Euclidean distance**:

\[
d(x, x') = \sqrt{\sum_{i=1}^{d}(x_i - x'_i)^2}
\]

---

## ⚙️ Algorithm Steps
1. Store the training data  
2. Compute distances between test point and all training points  
3. Select the *k* nearest neighbors  
4. Aggregate their labels to make a prediction  

---

## 🧠 Key Characteristics
- No explicit training phase
- Sensitive to feature scaling
- Computationally expensive at inference time

---

## 🚧 Limitations
- Slow for large datasets
- Memory intensive
- Performance depends heavily on choice of *k*
