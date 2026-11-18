# Intelligent Systems — ML Mini-Projects

Academic projects developed for the **Intelligent Systems** course — CIn/UFPE.

---

## Problem Statement

On April 15, 1912, the RMS Titanic sank after colliding with an iceberg, killing 1,502 of the 2,224 passengers and crew aboard. Survival was far from random — factors such as passenger class, sex, age, cabin allocation, and family size played a measurable role in who made it to a lifeboat.

The central question driving these projects is:

> **Can we reliably predict whether a passenger survived the Titanic disaster — and uncover natural passenger profiles — based solely on demographic and travel data?**

Beyond predictive accuracy, each mini-project investigates a distinct machine learning paradigm under strict leak-free methodology:

- **Decision Tree** — which features, split rules, pruning strategies, and tree ensembles most clearly separate survivors from non-survivors?
- **Multi-Layer Perceptron (MLP)** — how do attribute encoding, Min-Max scaling, leak-free repetitive class balancing, activation functions, optimizers, and L2 regularization affect neural network generalization?
- **Unsupervised Clustering** — after **strictly removing the `Survived` target**, do natural passenger groupings emerge via K-Means, K-Prototypes, and DBSCAN, how can Decision Trees extract human-readable rules for each group, and do those groups correlate *post-hoc* with historical survival rates?

---

## Team

| Name | Email |
|------|-------|
| Andreywid Yago Lima de Souza | ayls@cin.ufpe.br |
| Artur Vinicius Pereira Fernandes | arturvpf3@gmail.com |
| Felipe Mateus Falcao Barreto | fmfb@cin.ufpe.br |
| Joao Pedro Mafaldo de Paula | jpmp@cin.ufpe.br |
| Matheus Ayres dos Santos | mas14@cin.ufpe.br |

---

## Repository Structure

```text
.
├── decision-tree/
│   ├── Decision_Tree.ipynb             # Decision Tree, Pruning (ccp_alpha), Random Forest, Gradient Boosting
│   └── submission_decision_tree.csv    # Kaggle test predictions (418 passengers)
├── mlp/
│   ├── MLP.ipynb                       # Multi-Layer Perceptron, Class Balancing, Grid Search & Optuna
│   └── submission_mlp.csv              # Kaggle test predictions (418 passengers)
├── unsupervised-clustering/
│   └── Unsupervised.ipynb              # K-Means, K-Prototypes, DBSCAN & Decision Tree Rule Extraction
├── data/
│   ├── train.csv                       # Titanic labeled dataset (891 passengers)
│   ├── test.csv                        # Titanic Kaggle test dataset (418 passengers)
│   └── gender_submission.csv           # Baseline submission example
├── docs/
│   ├── Mini-projeto-arvoresdecisao2025.pdf
│   ├── Mini-projeto-MLP2025.pdf
│   └── Mini-projeto-Kmeans-K-prototypes-DBSCAN.pdf
└── requirements.txt                    # Python dependencies
```

---

## Dataset

All three projects use the **Titanic: Machine Learning from Disaster** dataset, publicly available on Kaggle:

[Titanic Dataset — Kaggle](https://www.kaggle.com/competitions/titanic)

The labeled dataset (`train.csv`) contains demographic and travel records for **891 passengers** (`Survived`: `0` = No, `61.62%`; `1` = Yes, `38.38%`), alongside **418 unlabeled test passengers** (`test.csv`).

| Feature | Description |
|---------|-------------|
| `Pclass` | Ticket socioeconomic class (`1` = 1st, `2` = 2nd, `3` = 3rd) |
| `Sex` | Passenger sex (`male`, `female`) |
| `Age` | Passenger age in years |
| `SibSp` | Number of siblings/spouses aboard |
| `Parch` | Number of parents/children aboard |
| `Fare` | Passenger ticket fare (£) |
| `Cabin` | Cabin identifier (mapped to binary `Has_Cabin`) |
| `Embarked` | Port of embarkation (`C` = Cherbourg, `Q` = Queenstown, `S` = Southampton) |

---

## Projects

### 1. Decision Tree & Tree Ensembles

**Notebook:** `decision-tree/Decision_Tree.ipynb`  
**Spec:** `docs/Mini-projeto-arvoresdecisao2025.pdf`

Explores single Decision Trees (pre-pruning and cost-complexity post-pruning) and tree ensembles (Random Forest and Gradient Boosting) for binary survival classification.

**Leak-Free Preprocessing & Split**
- Stratified **50% Train (445) / 25% Validation (223) / 25% Test (223)** split performed *before* imputation, scaling, or oversampling
- Identifier columns (`PassengerId`, `Name`, `Ticket`) removed; `Cabin` transformed into binary `Has_Cabin` (`0`/`1`)
- Missing `Age` imputed using median grouped by `Sex` and `Pclass` computed exclusively on `X_train`; `Embarked` filled with `X_train` mode (`S`); `Fare` imputed by `Pclass` median
- Engineered `Relatives = SibSp + Parch + 1`; binary encoding for `Sex` and One-Hot Encoding (`1-of-N`) for `Embarked`
- `MinMaxScaler` fitted strictly on `X_train` (plus unscaled `X_train` preserved for human-readable split thresholds)

**Manual Experiments Progression**

| Exp | Model & Configuration | Train Acc | Val Acc | Val F1 | Val ROC-AUC |
|-----|-----------------------|-----------|---------|--------|-------------|
| 0 | Decision Tree (`gini`, `depth=5`, `split=10`, `leaf=4`, `sqrt`) | 79.33% | 79.82% | 75.41% | 86.17% |
| 1 | Decision Tree (`entropy`, `depth=5`, `split=10`, `leaf=4`, `sqrt`) | 83.15% | 81.17% | 74.39% | 84.31% |
| 2 | Decision Tree (`entropy`, `depth=8`, `split=20`, `leaf=4`, `sqrt`) | 87.87% | 82.96% | 78.16% | 86.50% |
| 3 | Decision Tree (`entropy`, `depth=8`, `split=10`, `leaf=4`, `sqrt`) | 85.62% | 81.61% | 75.74% | 84.55% |
| 4 | Shallow Tree (`entropy`, `depth=4`, `split=10`, `leaf=4`, `sqrt`) | 75.51% | 73.54% | 70.65% | 84.54% |
| 5 | Full Pre-Pruned Tree (`max_features=None`, `max_leaf_nodes=14`, `min_imp=0.003`) | 84.27% | 78.03% | 73.22% | 83.36% |
| 6 | Cost-Complexity Post-Pruning (`ccp_alpha=0.00584`) | 91.46% | 83.86% | 77.50% | 81.60% |
| 7 | Decision Tree + Repetitive Oversampling (`RandomOverSampler` on Train) | 84.85% | 81.61% | 74.85% | 82.21% |
| 8 | Random Forest (`100` trees, `entropy`, `depth=8`, `leaf=4`) | 88.31% | 82.06% | 77.27% | **91.23%** |
| 9 | Gradient Boosting (`100` trees, `lr=0.05`, `depth=4`, `split=20`) | 89.89% | **85.20%** | **79.50%** | 90.57% |

**Hyperparameter Tuning & Held-Out Test Evaluation**
- **Grid Search (`PredefinedSplit`):** `RandomForestClassifier` achieved **87.00% Validation Accuracy** and **79.82% Held-Out Test Accuracy** (**82.54% Test ROC-AUC**).
- **Optuna (`TPESampler`):** `GradientBoostingClassifier` achieved **86.55% Validation Accuracy** and **78.48% Held-Out Test Accuracy** (**79.81% Test ROC-AUC**), outperforming the single Decision Tree on unseen test passengers while preserving clear feature-importance alignment (`Sex`, `Fare`, `Pclass`, `Age`, `Has_Cabin`, `Relatives`).

---

### 2. Multi-Layer Perceptron (MLP)

**Notebook:** `mlp/MLP.ipynb`  
**Spec:** `docs/Mini-projeto-MLP2025.pdf`

Investigates neural network architectures (`MLPClassifier`) trained via backpropagation, following strict symbolic encoding (`1-of-N`), `MinMaxScaler([0, 1])`, and **post-split repetitive minority oversampling (`RandomOverSampler` + shuffling)** to eliminate data contamination across Train (50%), Validation (25%), and Test (25%).

**Manual Experiments Progression**

| Exp | Architecture | Activation | Optimizer | Regularization / Setup | Val Acc | Val Recall | Val F1 |
|-----|--------------|------------|-----------|------------------------|---------|------------|--------|
| 0A | `(100,)` | ReLU | Adam | `alpha=1e-4` (Unbalanced Train) | 83.41% | 72.09% | 77.02% |
| 0B | `(100,)` | ReLU | Adam | `alpha=1e-4` (Balanced Train) | 83.41% | 77.91% | 78.36% |
| 1 | `(16,)` | ReLU | Adam | Compact single hidden layer | 82.51% | 80.23% | 77.97% |
| 2A | `(50, 25)` | ReLU | Adam | 2 hidden layers (`alpha=1e-4`) | **84.75%** | 80.23% | **80.23%** |
| 2B | `(64, 32, 16)` | ReLU | Adam | 3 hidden layers (`alpha=1e-4`) | 83.86% | 75.58% | 78.31% |
| 3A | `(50, 25)` | Logistic | Adam | Sigmoid activation | 82.06% | 80.23% | 77.53% |
| 3B | `(50, 25)` | Tanh | Adam | Hyperbolic tangent activation | 83.86% | 77.91% | 78.82% |
| 4A | `(50, 25)` | Tanh | SGD | Nesterov momentum + adaptive `lr=0.02`, `L2=1e-3` | 83.41% | 79.07% | 78.61% |
| 4B | `(50, 25)` | ReLU | SGD | High constant `lr=0.5`, `L2=1e-2` | 83.41% | 77.91% | 78.36% |
| 5A | `(50, 25)` | ReLU | Adam | Strong L2 (`alpha=0.05`, `lr=0.002`) | 83.86% | 75.58% | 78.31% |
| 5B | `(64, 32)` | ReLU | Adam | L2 (`alpha=0.02`) + `early_stopping=True` | 81.61% | **81.40%** | 77.35% |

**Hyperparameter Tuning & Held-Out Test Evaluation**
- **Grid Search:** `(64, 32)` with `ReLU`, `Adam`, `alpha=0.01`, `lr=0.005` reached **85.65% Validation Accuracy** (**80.95% Val F1**).
- **Optuna (120 trials):** `(112, 64)` with `Logistic`, `Adam`, `alpha=0.0073`, `lr=0.0159` achieved **86.10% Validation Accuracy** and **80.27% Held-Out Test Accuracy** (**82.33% Test ROC-AUC**).

---

### 3. Unsupervised Clustering

**Notebook:** `unsupervised-clustering/Unsupervised.ipynb`  
**Spec:** `docs/Mini-projeto-Kmeans-K-prototypes-DBSCAN.pdf`

Treats all **891 passengers** as a strictly unsupervised problem (`Survived` is immediately removed upon loading) to discover natural passenger segments, evaluate cluster geometry, and extract human-readable rules via Decision Trees.

**Algorithms & Comparative Results**

| Algorithm | Paradigm & Feature Space | Selected Parameters | Clusters (% Noise) | Native Silhouette | Davies-Bouldin | Min Inter / Mean Intra |
|-----------|--------------------------|---------------------|--------------------|-------------------|----------------|------------------------|
| **K-Means** | Partitional (Normalized numerical/ordinal) | `k = 4` (majority vote across 5 metrics) | 4 (`0.00%`) | `0.6085` | `0.6770` | `2.78` |
| **K-Prototypes** | Partitional (Mixed: numerical + categorical) | `k = 10`, `gamma = 1.0` | 10 (`0.00%`) | `0.5469` | `1.2086` | `2.66` |
| **DBSCAN** | Density-based (Normalized 9D space) | `eps = 0.55`, `min_samples = 15` | 10 (`1.23%` noise) | **`0.6315`** | **`0.5679`** | **`3.64`** |

**Methodology & Key Findings**
1. **Multi-Metric `k` Selection:** Optimal `k` chosen via majority vote over **Elbow (`KneeLocator`), Calinski-Harabasz, Davies-Bouldin, Silhouette, and X-Means BIC**.
2. **Cluster Geometry:** Intra-cluster cohesion and pairwise inter-cluster distance heatmaps computed across all formed groups.
3. **Real-Unit Profiling:** K-Means (`k = 4`) naturally separates **large 3rd-class families** (`Grupo 1`, avg. `6.67` relatives, median age `11`), **1st-class cabin holders** (`Grupo 2`, avg. fare `£84.15`, `81.5%` cabin), **2nd-class travelers** (`Grupo 3`), and **solo/small-family 3rd-class passengers** (`Grupo 4`), while K-Prototypes and DBSCAN further partition passengers by sex, port, and cabin — isolating `11` atypical high-fare/rare-combination passengers as noise (`-1`).
4. **Decision Tree Rule Extraction:** Shallow `DecisionTreeClassifier` models trained on cluster labels (`Grupo 1..k`) achieve **99.5%–100% fidelity**, yielding concise `IF ... THEN` rules in original passenger units.
5. **Post-Hoc External Validation:** Without ever using `Survived` during clustering, discovered groups span historical survival rates from **7.69%–13.79%** (3rd-class men and large 3rd-class families) up to **91.03%–97.22%** (1st/2nd-class women with registered cabins).

---

## Getting Started

### Prerequisites

- Python 3.10+
- Jupyter Notebook, JupyterLab, VS Code, or Google Colab

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run any notebook locally

```bash
jupyter notebook decision-tree/Decision_Tree.ipynb
jupyter notebook mlp/MLP.ipynb
jupyter notebook unsupervised-clustering/Unsupervised.ipynb
```

All three notebooks detect `../data/` automatically when run locally and also support Google Colab out of the box.

---

## Tech Stack

- **Python 3**
- **Pandas & NumPy** — data manipulation and numerical operations
- **Matplotlib & Seaborn** — statistical and 3D visualizations
- **Scikit-learn** — preprocessing, Decision Trees, Ensembles, MLP, K-Means, DBSCAN, PCA, and evaluation metrics
- **Imbalanced-learn** — leak-free minority class oversampling (`RandomOverSampler`)
- **Optuna** — Bayesian hyperparameter optimization (`TPESampler`)
- **KModes & Kneed** — `KPrototypes` for mixed-type clustering and automated elbow detection (`KneeLocator`)

---

## License

Academic project — for educational use only. CIn/UFPE, 2025.
