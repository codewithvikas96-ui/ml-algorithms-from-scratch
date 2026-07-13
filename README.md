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
| 6 | SVM | Classification | Subgradient Descent (Pegasos-style) |
| 7 | Bagging Classifier | Classification | Bootstrap Aggregating + Majority Vote |

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

## 6. Support Vector Machine (SVM)

**File:** `notebooks/SVM.ipynb`

### Concept

SVM finds the **optimal hyperplane** that separates two classes with the **maximum margin** — the widest possible gap between the nearest points of each class (called support vectors). Instead of just finding *a* decision boundary that works, SVM finds the one that is *furthest* from both classes simultaneously, making it more robust to new data.

This implementation uses **subgradient descent** (Pegasos-style) to minimize a regularized hinge loss — a scalable alternative to the exact quadratic programming (QP) solver used in sklearn's SVC.

### The Math

**Objective function (regularized hinge loss):**

$$L(w, b) = \lambda\|w\|^2 + \frac{1}{n}\sum_{i=1}^{n}\max\left(0,\ 1 - y_i(w \cdot x_i + b)\right)$$

The first term is L2 regularization (controls margin width), the second is the hinge loss (penalizes misclassified points and points inside the margin).

**Subgradient update rules:**

For a correctly classified point where $y_i(w \cdot x_i + b) \geq 1$ (outside the margin):

$$w := w - \alpha \cdot 2\lambda w$$

For a misclassified or margin-violating point where $y_i(w \cdot x_i + b) < 1$:

$$w := w - \alpha \cdot (2\lambda w - y_i x_i)$$

$$b := b - \alpha \cdot (-y_i)$$

**Decaying learning rate (stabilizes convergence):**

$$\alpha_t = \frac{\alpha_0}{1 + \lambda \cdot t}$$

**Decision function (signed distance from hyperplane):**

$$f(x) = w \cdot x + b \quad \Rightarrow \quad \hat{y} = \text{sign}(f(x))$$

### Step-by-Step Implementation

1. Convert labels to $\{-1, +1\}$ internally — SVM's hinge loss requires this label convention. **Important:** the original `y` array passed in is not mutated (it's a local reassignment inside `fit()`), so predictions come back as $\{-1, +1\}$. Compare against a converted `y_true`, not the raw `y` from `make_blobs` (which is $\{0, 1\}$) — doing so gives a false ~50% accuracy regardless of model quality.
2. Initialize `w = zeros`, `b = 0`.
3. For each epoch, shuffle indices (avoids cyclical bias), compute `current_lr` with decay.
4. For each sample: check if it satisfies the margin condition. If yes → only regularize `w`. If no → update both `w` and `b` using the hinge subgradient.
5. After each epoch, compute and store the full hinge loss via `_hinge_loss()` for plotting convergence.
6. `predict()` returns `np.sign(X @ w + b)`.
7. `decision_function()` returns the raw signed distance — useful for identifying approximate support vectors (points closest to the hyperplane).

### Key Properties

- Labels **must** be $\{-1, +1\}$, not $\{0, 1\}$ — the hinge loss formula `max(0, 1 - y*(w·x + b))` breaks silently with binary labels.
- The pure Pegasos schedule $\alpha_t = \frac{1}{\lambda t}$ can blow up for small `lambda_param` values — the modified schedule $\frac{\alpha_0}{1 + \lambda t}$ is more stable in practice.
- This is a **linear SVM** — for non-linearly separable data, a kernel (RBF, polynomial) would be needed, which requires the dual formulation.
- Subgradient descent gives approximate support vectors, not exact ones (unlike QP). Points within ~15th percentile of `|decision_function|` are used as proxies.

---

## 7. Bagging Classifier (Bootstrap Aggregating)

**File:** `notebooks/BaggingClassifier.ipynb`

### Concept

Bagging (Bootstrap Aggregating) is an **ensemble method** that reduces variance by training multiple independent models on different random subsets of the training data, then combining their predictions via **majority vote**.

The key insight: individual decision trees have high variance — they overfit to whatever training data they see. If you train many trees on slightly different versions of the data (bootstrap resamples) and average their opinions, the variance cancels out while the bias stays roughly the same.

### The Math

**Bootstrap resampling** — for a dataset of $n$ samples, draw $n$ samples *with replacement*:

$$D_i = \{(x_j, y_j) \mid j \sim \text{Uniform}\{1, \ldots, n\}, \text{ with replacement}\}$$

On average, each bootstrap sample contains ~63.2% of unique original samples — the remaining ~36.8% are the **Out-of-Bag (OOB)** samples for that tree.

**Final prediction (majority vote):**

$$\hat{y} = \text{mode}\left(\hat{y}_1(x),\ \hat{y}_2(x),\ \ldots,\ \hat{y}_T(x)\right)$$

**OOB Score** (free internal validation, no separate test set needed):

$$\text{OOB Score} = \frac{1}{n}\sum_{i=1}^{n} \mathbf{1}\left[\hat{y}_{\text{oob},i} = y_i\right]$$

where $\hat{y}_{\text{oob},i}$ is the majority vote from only the trees that did **not** see sample $i$ in their bootstrap resample.

### Step-by-Step Implementation

1. For each of `n_estimators` trees:
   - Draw a bootstrap resample of size $n$ with replacement using `rng.choice(..., replace=True)`.
   - Track which original indices were **not** selected → OOB mask for this tree.
   - Train a `DecisionTreeClassifier` on the resampled data.
   - For all OOB indices, collect this tree's predictions into a running vote list.
2. After all trees are trained, for each sample: take the majority vote across all trees that had it as OOB → compute OOB accuracy (`oob_score_`).
3. `predict()` collects predictions from all trees, transposes the array (shape: `[n_estimators, n_samples]` → `[n_samples, n_estimators]`), then returns the majority class per sample using `Counter`.
4. `predict_proba()` returns the fraction of trees voting for each class per sample — a soft probability estimate.

### Key Properties

- Each tree is trained **independently** and in **full depth** (or up to `max_depth`) — unlike Random Forest, Bagging uses the full feature set at every split (no feature subsampling per split).
- OOB score is a nearly unbiased estimate of generalization error — comparable to cross-validation but computed for free during training, without a separate validation set.
- Bagging reduces **variance**, not **bias** — it won't fix an underfitting model, only an overfitting one.
- The difference from **Random Forest**: Random Forest adds feature subsampling at each split on top of bootstrap sampling, which further decorrelates the trees and usually gives better performance.

| | Bagging | Random Forest |
|---|---|---|
| Bootstrap sampling | ✅ | ✅ |
| Feature subsampling per split | ❌ | ✅ (`sqrt(n_features)`) |
| Tree correlation | Higher | Lower |
| Typical performance | Good | Better |

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
    ├── KNNClassifier.ipynb
    ├── SVM.ipynb
    └── BaggingClassifier.ipynb
```

## Tech Stack

- **Python 3**
- **NumPy** — all core math (distances, matrix operations, gradients)
- **scikit-learn** — used *only* for `StandardScaler` (feature scaling), not for any model
- **Matplotlib** — visualizing predictions and loss curves

--- 

## Key Takeaways from This Project

- **OLS vs. GD**: OLS gives an exact solution instantly for small datasets but doesn't scale; GD scales to large datasets but needs tuning (`learning_rate`, `epochs`) and feature scaling.
- **Linear vs. Logistic Regression**: The gradient formulas for both look almost identical — the only real difference is that Logistic Regression passes its linear output through a sigmoid function and uses Cross-Entropy loss instead of MSE.
- **KNN's tradeoff**: No training cost, but expensive predictions, and heavily reliant on proper feature scaling since it's purely distance-based.
- **SVM's margin intuition**: Unlike Logistic Regression which just finds *a* boundary that separates classes, SVM finds the boundary with *maximum margin* — making it more robust. The hinge loss only penalizes points that are misclassified or inside the margin, completely ignoring points already on the correct side.
- **Bagging vs. a single tree**: A single decision tree overfits because it memorizes the training data. Bagging forces each tree to see a slightly different version of the data (bootstrap resample), so their errors don't all point in the same direction — the majority vote cancels out individual mistakes and reduces variance without touching bias.
- **Bagging vs. Random Forest**: Bagging uses the full feature set at every split; Random Forest additionally subsamples features per split, further decorrelating the trees. That one extra step is what usually makes Random Forest outperform plain Bagging.
- Implementing these from scratch makes the role of the **cost function**, **gradient**, and **learning rate** click in a way that just calling `model.fit()` never does. And implementing ensemble methods from scratch shows *why* combining weak learners works — it's not magic, it's variance reduction through averaging.


## Author

**Vikas Ajay Vishwakarma**
BSc IT student | Aspiring Data Scientist
GitHub: [@codewithvikas96-ui](https://github.com/codewithvikas96-ui)
