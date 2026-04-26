# Machine Learning From Scratch

This repository contains **clean, well-documented implementations of core Machine Learning algorithms from scratch**, using only fundamental Python libraries such as `NumPy` and `Matplotlib`.

The goal is **deep understanding**, not library usage.

---

## Motivation

Most ML practitioners use high-level libraries without understanding what happens underneath.  
This project is designed to:

- Understand ML algorithms mathematically
- Implement them step-by-step
- Analyze their strengths and weaknesses
- Compare them with standard library implementations

This repository is also intended as **educational material** for students and researchers.

---

## What "From Scratch" Means

- No `scikit-learn`, `xgboost`, `lightgbm` for model implementations  
- Only `numpy`, `math`, `matplotlib`  
- Explicit loss functions, gradients, and optimization
- `scikit-learn` used only in benchmarks for comparison

---

## Interactive UI

Explore every algorithm visually with the **Streamlit app**:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Features:
- Select any algorithm from the sidebar
- Adjust hyperparameters with sliders and see results update live
- View decision boundaries, loss curves, cluster plots, and PCA projections
- Read the mathematical theory for each algorithm
- View the from-scratch source code

---

## Implemented Algorithms

### Supervised Learning

| Algorithm | Directory | Key Files |
|-----------|-----------|-----------|
| Linear Regression | `algorithms/supervised/linear_regression/` | model.py, loss.py, optimizer.py, demo.py |
| Ridge Regression (L2) | `algorithms/supervised/linear_regression/` | model.py (RidgeRegression class) |
| Lasso Regression (L1) | `algorithms/supervised/linear_regression/` | model.py (LassoRegression class) |
| Logistic Regression | `algorithms/supervised/logistic_regression/` | model.py, loss.py, optimizer.py, demo.py |
| Regularized Logistic (L2) | `algorithms/supervised/logistic_regression/` | model.py (RegularizedLogisticRegression class) |
| k-Nearest Neighbors | `algorithms/supervised/knn/` | model.py, distance.py, demo.py |
| Naive Bayes | `algorithms/supervised/naive_bayes/` | model.py, demo.py |
| Decision Tree | `algorithms/supervised/decision_tree/` | model.py, demo.py |
| Neural Network (MLP) | `algorithms/supervised/neural_network/` | model.py, demo.py |

### Unsupervised Learning

| Algorithm | Directory | Key Files |
|-----------|-----------|-----------|
| K-Means Clustering | `algorithms/unsupervised/kmeans/` | model.py, demo.py |
| Hierarchical Clustering | `algorithms/unsupervised/hierarchical_clustering/` | model.py, demo.py |
| Principal Component Analysis (PCA) | `algorithms/unsupervised/pca/` | model.py, demo.py |

### Ensemble Methods

| Algorithm | Directory | Key Files |
|-----------|-----------|-----------|
| Bagging | `algorithms/ensemble/bagging/` | model.py, demo.py |
| Random Forest | `algorithms/ensemble/random_forest/` | model.py, demo.py |
| Boosting (AdaBoost) | `algorithms/ensemble/boosting/` | model.py, demo.py |

---

## Utilities (From Scratch)

| Utility | File | Functions / Classes |
|---------|------|---------------------|
| Train/Test Split | `utils/data_split.py` | `train_test_split`, `k_fold_split` |
| Feature Scaling | `utils/scalers.py` | `StandardScaler`, `MinMaxScaler` |
| Metrics | `utils/metrics.py` | `accuracy`, `precision`, `recall`, `f1_score`, `confusion_matrix`, `mean_squared_error`, `r_squared`, `silhouette_score` |

---

## Project Structure

```
ML-From-Scratch/
|
|-- app.py                  # Streamlit interactive UI
|
|-- algorithms/
|   |-- supervised/
|   |   |-- linear_regression/    # Linear, Ridge (L2), Lasso (L1)
|   |   |-- logistic_regression/  # Standard + L2 regularized
|   |   |-- knn/                  # Instance-based learning
|   |   |-- naive_bayes/          # Gaussian Naive Bayes
|   |   |-- decision_tree/        # Entropy-based classifier
|   |   |-- neural_network/       # MLP with backpropagation
|   |
|   |-- unsupervised/
|   |   |-- kmeans/                    # Centroid-based clustering
|   |   |-- hierarchical_clustering/   # Agglomerative clustering
|   |   |-- pca/                       # Dimensionality reduction
|   |
|   |-- ensemble/
|       |-- bagging/          # Bootstrap aggregating
|       |-- random_forest/    # Bagging + random feature subsets
|       |-- boosting/         # AdaBoost with decision stumps
|
|-- utils/
|   |-- data_split.py    # train_test_split, k_fold_split
|   |-- scalers.py       # StandardScaler, MinMaxScaler
|   |-- metrics.py       # accuracy, precision, recall, f1, etc.
|
|-- math/
|   |-- linear_algebra.md    # Vectors, matrices, eigenvalues
|   |-- probability.md       # Bayes theorem, distributions, MLE
|   |-- optimization.md      # Gradient descent, convexity, convergence
|
|-- benchmarks/              # Scratch vs scikit-learn comparisons
|-- requirements.txt
|-- README.md
```

---

## Mathematical Foundations

- [Linear Algebra](math/linear_algebra.md) - Vectors, matrices, eigendecomposition
- [Probability Theory](math/probability.md) - Bayes' theorem, distributions, MLE, entropy
- [Optimization Techniques](math/optimization.md) - Gradient descent, convexity, convergence

All derivations are explained intuitively in the `math/` directory.

---

## Benchmarks

Each algorithm is compared against its `scikit-learn` counterpart in the `benchmarks/` directory:

```bash
cd benchmarks
python benchmark_linear_regression.py
python benchmark_logistic_regression.py
python benchmark_knn.py
python benchmark_naive_bayes.py
python benchmark_decision_tree.py
python benchmark_kmeans.py
python benchmark_random_forest.py
python benchmark_adaboost.py
```

---

## Getting Started

```bash
# Clone the repository
git clone https://github.com/joytun-tonny/ML-From-Scratch.git
cd ML-From-Scratch

# Install dependencies
pip install -r requirements.txt

# Launch the interactive UI
streamlit run app.py

# Or run any algorithm demo directly
cd algorithms/supervised/linear_regression
python demo.py
```

---

## Educational Value

This repository is structured to:

- Help students learn ML fundamentals
- Support teaching and mentoring
- Serve as a research-ready portfolio

Each algorithm folder contains:
- `model.py` - Clean from-scratch implementation
- `demo.py` - Working example with visualizations
- `README.md` - Mathematical theory and intuition

---

## Author

**Joytun Nessa Tonny**  
AI & Data Science Researcher  
GitHub: https://github.com/joytun-tonny
