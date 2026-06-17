![Python](https://img.shields.io/badge/Python-3.13-3776AB?style=flat-square&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-Vectorized-013243?style=flat-square&logo=numpy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Preprocessing%20Only-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-11557C?style=flat-square&logo=plotly&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-F37626?style=flat-square&logo=jupyter&logoColor=white)
![From Scratch](https://img.shields.io/badge/Built-From%20Scratch-success?style=flat-square)
![Algorithms](https://img.shields.io/badge/Algorithms-5-blue?style=flat-square)
![ML](https://img.shields.io/badge/Machine%20Learning-Supervised-orange?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)
![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=flat-square)


# ML Algorithms from Scratch

Implementation of core Machine Learning algorithms using **only NumPy** — no `sklearn` models, no shortcuts. The goal of this repository is to understand *what's actually happening* inside these algorithms: the math, the gradients, and the optimization, instead of treating them as black boxes.

Each algorithm is implemented as a Python class with a familiar `fit()` / `predict()` API (similar to scikit-learn), tested on small toy datasets, and verified against expected behavior.

## Why this repo exists

Most tutorials teach you to call `model.fit(X, y)` from a library. This repo goes one level deeper — every algorithm here is built from the underlying equations, so the linear algebra, calculus, and optimization logic is fully visible and understood, not hidden behind a library call.

## Algorithms Implemented

| # | Algorithm | Type | Method |
|---|-----------|------|--------|
| 1 | Linear Regression (OLS) | Regression | Closed-form Normal Equation |
| 2 | Linear Regression (GD) | Regression | Gradient Descent |
| 3 | Logistic Regression | Classification | Gradient Descent |
| 4 | KNN Regressor | Regression | Distance-based, Lazy Learning |
| 5 | KNN Classifier | Classification | Distance-based, Lazy Learning |

---

## 1. Linear Regression — Ordinary Least Squares (OLS)

**File:** `notebooks/LinearRegressionOLS.ipynb`

### Concept

Linear Regression models the relationship between input features and a continuous target as a straight line (or hyperplane in higher dimensions):

$$\hat{y} = X\beta$$

where $\beta$ includes both the intercept and coefficients. Rather than iteratively improving the parameters, OLS solves for the **exact optimal** parameters in one shot using calculus — by minimizing the Sum of Squared Errors (SSE) and solving where its derivative is zero.

### The Math

Cost function (Sum of Squared Errors):

$$J(\beta) = (y - X\beta)^T(y - X\beta)$$

Taking the derivative with respect to $\beta$ and setting it to zero gives the **Normal Equation**:

$$\beta = (X^TX)^{-1}X^Ty$$

Here, $X$ is augmented with a column of ones to account for the intercept term.

### Step-by-Step Implementation

1. Convert `X` and `y` to NumPy arrays.
2. Add a bias column of ones to `X` → `X_bias = [1 | X]`.
3. Compute $\beta = (X_{bias}^T X_{bias})^{-1} X_{bias}^T y$ using `np.linalg.inv()`.
4. Split $\beta$ into `intercept` (first value) and `coefficients` (remaining values).
5. Predict using $\hat{y} = X\beta_{coef} + \beta_{intercept}$.

### Key Properties

- No learning rate or epochs needed — solved in a single matrix computation.
- Computationally expensive for large feature counts, since matrix inversion is $O(n^3)$.
- Fails (or becomes unstable) when $X^TX$ is non-invertible — e.g. with highly correlated (multicollinear) features.

---

## 2. Linear Regression — Gradient Descent (GD)

**File:** `notebooks/LinearRegressionGD.ipynb`

### Concept

Instead of solving for $\beta$ directly, Gradient Descent starts with random (zero) parameters and **iteratively nudges them** in the direction that reduces error, using the gradient (slope) of the cost function.

### The Math

Prediction:

$$\hat{y} = Xw + b$$

Mean Squared Error cost function:

$$J(w,b) = \frac{1}{2n}\sum_{i=1}^{n}(\hat{y}_i - y_i)^2$$

Gradients (partial derivatives of cost with respect to weights and bias):

$$\frac{\partial J}{\partial w} = \frac{1}{n}X^T(\hat{y} - y)$$

$$\frac{\partial J}{\partial b} = \frac{1}{n}\sum_{i=1}^{n}(\hat{y}_i - y_i)$$

Parameter update rule (where $\alpha$ is the learning rate):

$$w := w - \alpha \frac{\partial J}{\partial w}$$

$$b := b - \alpha \frac{\partial J}{\partial b}$$

### Step-by-Step Implementation

1. Initialize `weights` as zeros (one per feature) and `bias` as 0.
2. For each epoch:
   - Compute predictions: `y_pred = X @ weights + bias`
   - Compute error: `error = y_pred - y`
   - Compute gradients `dw` and `db` from the error.
   - Update `weights` and `bias` using the learning rate.
   - Record the MSE loss for that epoch in `loss_history`.
3. After training, `predict()` simply applies `X @ weights + bias`.

### Key Properties

- Requires **feature scaling** (`StandardScaler`) — without it, features on different scales (e.g. square footage vs. number of rooms) cause unstable or slow convergence.
- Requires tuning `learning_rate` and `epochs`.
- Scales well to large datasets and high-dimensional data, unlike OLS.
- `loss_history` lets you visualize convergence — the loss should decrease and flatten out as training progresses.

> **Note:** The loss must be computed as `(1 / (2 * n_samples)) * np.sum(error ** 2)`. Watch the parentheses — `(1 / 2 * n_samples)` is a classic Python precedence bug that multiplies by `n_samples` instead of dividing by it, silently corrupting the loss curve (though not the gradients, which are computed independently).

---

## 3. Logistic Regression

**File:** `notebooks/LogisticRegression.ipynb`

### Concept

Logistic Regression is used for **binary classification**. It works like linear regression, but instead of outputting a raw number, it passes the result through a **sigmoid function** to squash it into a probability between 0 and 1.

### The Math

Linear combination:

$$z = Xw + b$$

Sigmoid (logistic) function:

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

Predicted probability:

$$\hat{y} = \sigma(z)$$

Cost function — **Binary Cross-Entropy (Log Loss)**:

$$J(w,b) = -\frac{1}{n}\sum_{i=1}^{n}\left[y_i\log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i)\right]$$

Gradients (notice these take the *same form* as linear regression's gradients — only $\hat{y}$ is now sigmoid-based):

$$\frac{\partial J}{\partial w} = \frac{1}{n}X^T(\hat{y} - y)$$

$$\frac{\partial J}{\partial b} = \frac{1}{n}\sum_{i=1}^{n}(\hat{y}_i - y_i)$$

Final classification (using a threshold, typically 0.5):

$$\text{class} = \begin{cases} 1 & \text{if } \hat{y} \geq 0.5 \\ 0 & \text{if } \hat{y} < 0.5 \end{cases}$$

### Step-by-Step Implementation

1. Initialize `weights` as zeros and `bias` as 0.
2. For each epoch:
   - Compute `z = X @ weights + bias`.
   - Apply `_sigmoid(z)` (with `np.clip` on `z` to `[-500, 500]` to avoid `np.exp` overflow).
   - Compute gradients `dw`, `db` from `(y_pred - y)`.
   - Update `weights` and `bias`.
   - Compute and store Binary Cross-Entropy loss (with a small `epsilon` added inside the `log()` calls to avoid `log(0)`).
3. `predict_proba()` returns the raw sigmoid probability.
4. `predict()` thresholds that probability (default 0.5) into a 0/1 class label.

### Key Properties

- Despite the name, this is a **classification** algorithm, not regression.
- The decision boundary is linear in the feature space.
- Like Linear Regression GD, this benefits from feature scaling for stable, faster convergence.
- The sigmoid output is interpretable as a genuine probability, not just a score.

---

## 4. K-Nearest Neighbors — Regressor

**File:** `notebooks/KNNRegressor.ipynb`

### Concept

KNN is a **lazy learner** — it doesn't learn parameters during training at all. It simply memorizes the training data. At prediction time, it finds the `k` closest training points to the new sample and **averages their target values**.

### The Math

Euclidean distance between two points:

$$d(x_1, x_2) = \sqrt{\sum_{i=1}^{n}(x_{1i} - x_{2i})^2}$$

Prediction (average of the $k$ nearest neighbors' targets):

$$\hat{y} = \frac{1}{k}\sum_{i=1}^{k} y_i$$

### Step-by-Step Implementation

1. `fit()` just stores `X_train` and `y_train` — no actual training happens.
2. For each new sample, in `_predict_single()`:
   - Compute the Euclidean distance to every point in `X_train`.
   - Use `np.argsort()` to find the indices of the `k` smallest distances.
   - Retrieve the `y` values of those `k` nearest neighbors.
   - Return their mean as the prediction.
3. `predict()` simply loops `_predict_single()` over all samples in `X`.

### Key Properties

- No training phase — all the computational cost happens at prediction time, $O(n)$ per prediction.
- Extremely sensitive to feature scale — a feature in the thousands (e.g. square footage) would dominate distance over a feature like number of bedrooms unless scaled. This is why `StandardScaler` is applied before fitting.
- Choice of `k` controls the bias-variance tradeoff: small `k` → low bias, high variance (overfits to noise); large `k` → high bias, low variance (oversmooths).

---

## 5. K-Nearest Neighbors — Classifier

**File:** `notebooks/KNNClassifier.ipynb`

### Concept

Same lazy-learning idea as the KNN Regressor, but instead of averaging neighbor values, it takes a **majority vote** of the neighbors' class labels.

### The Math

Euclidean distance (same as above):

$$d(x_1, x_2) = \sqrt{\sum_{i=1}^{n}(x_{1i} - x_{2i})^2}$$

Prediction (majority class among $k$ nearest neighbors):

$$\hat{y} = \text{mode}(y_1, y_2, \dots, y_k)$$

### Step-by-Step Implementation

1. `fit()` stores `X_train` and `y_train`.
2. For each new sample, in `_predict_single()`:
   - Compute Euclidean distance to every training point.
   - Find indices of the `k` nearest neighbors via `np.argsort()`.
   - Retrieve their class labels.
   - Use `collections.Counter` to count label frequencies and return the most common one via `.most_common(1)`.
3. `predict()` loops this over all input samples.

### Key Properties

- Like the regressor version, it requires feature scaling and has no real "training" step.
- Ties in majority voting are broken by whichever label `Counter` encounters first — for production use, this is usually handled more carefully (e.g. weighting by inverse distance).
- Works naturally for multi-class problems too, not just binary.

---

## Repository Structure

```
ml-algorithms-from-scratch/
│
├── README.md
└── notebooks/
    ├── LinearRegressionOLS.ipynb
    ├── LinearRegressionGD.ipynb
    ├── LogisticRegression.ipynb
    ├── KNNRegressor.ipynb
    └── KNNClassifier.ipynb
```

## Tech Stack

- **Python 3**
- **NumPy** — all core math (distances, matrix operations, gradients)
- **scikit-learn** — used *only* for `StandardScaler` (feature scaling), not for any model
- **Matplotlib** — visualizing predictions and loss curves

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/codewithvikas96-ui/ml-algorithms-from-scratch.git
   cd ml-algorithms-from-scratch
   ```
2. Install dependencies:
   ```bash
   pip install numpy scikit-learn matplotlib jupyter
   ```
3. Launch Jupyter and open any notebook inside `notebooks/`:
   ```bash
   jupyter notebook
   ```

## Key Takeaways from This Project

- **OLS vs. GD**: OLS gives an exact solution instantly for small datasets but doesn't scale; GD scales to large datasets but needs tuning (`learning_rate`, `epochs`) and feature scaling.
- **Linear vs. Logistic Regression**: The gradient formulas for both look almost identical — the only real difference is that Logistic Regression passes its linear output through a sigmoid function and uses Cross-Entropy loss instead of MSE.
- **KNN's tradeoff**: No training cost, but expensive predictions, and heavily reliant on proper feature scaling since it's purely distance-based.
- Implementing these from scratch makes the role of the **cost function**, **gradient**, and **learning rate** click in a way that just calling `model.fit()` never does.

## Author

**Vikas Ajay Vishwakarma**
BSc IT student | Aspiring Data Scientist
GitHub: [@codewithvikas96-ui](https://github.com/codewithvikas96-ui)
