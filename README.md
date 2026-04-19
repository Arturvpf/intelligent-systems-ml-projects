# Intelligent Systems — ML Mini-Projects

Academic projects developed for the **Intelligent Systems** course — CIn/UFPE

---

## Problem Statement

On April 15, 1912, the RMS Titanic sank after colliding with an iceberg, killing 1,502 of the 2,224 passengers and crew aboard. Survival was far from random — factors such as passenger class, sex, age, and family size played a measurable role in who made it to a lifeboat.

The central question driving these projects is:

> **Can we reliably predict whether a passenger survived the Titanic disaster based solely on their demographic and travel data?**

Beyond prediction accuracy, each mini-project investigates a different facet of this problem:

- **Decision Tree** — which features and decision rules most clearly separate survivors from non-survivors?
- **MLP** — can a neural network learn non-linear survival patterns that simpler models miss?
- **Unsupervised Clustering** — without using survival labels, do natural passenger groupings emerge from the data, and do those groups correlate with survival rates?

---

## Team

| Name | Email |
|------|-------|
| Andreywid Yago Lima de Souza | ayls@cin.ufpe.br |
| Artur Vinicius Pereira Fernandes | avpf@cin.ufpe.br |
| Felipe Mateus Falcao Barreto | fmfb@cin.ufpe.br |
| Joao Pedro Mafaldo de Paula | jpmp@cin.ufpe.br |
| Matheus Ayres dos Santos | mas14@cin.ufpe.br |

---

## Repository Structure

```
.
├── decision-tree/
│   └── Decision_Tree.ipynb       # Decision Tree, Random Forest, Gradient Boosting
├── mlp/
│   └── MLP.ipynb                 # Multi-Layer Perceptron
├── unsupervised-clustering/
│   └── Unsupervised.ipynb        # K-Means, K-Prototypes, DBSCAN
├── data/
│   ├── train.csv                 # Titanic training set (891 passengers)
│   ├── test.csv                  # Titanic test set (418 passengers)
│   └── gender_submission.csv     # Baseline submission example
└── docs/
    ├── Mini-projeto-arvoresdecisao2025.pdf
    ├── Mini-projeto-MLP2025.pdf
    └── Mini-projeto-Kmeans-K-prototypes-DBSCAN.pdf
```

---

## Dataset

All three projects use the **Titanic: Machine Learning from Disaster** dataset, publicly available on Kaggle:

[Titanic Dataset — Kaggle](https://www.kaggle.com/competitions/titanic)

The dataset contains demographic and travel information for 891 passengers of the RMS Titanic. The target variable is `Survived` (0 = No, 1 = Yes).

| Feature | Description |
|---------|-------------|
| `Pclass` | Ticket class (1st, 2nd, 3rd) |
| `Sex` | Passenger sex |
| `Age` | Passenger age |
| `SibSp` | Number of siblings/spouses aboard |
| `Parch` | Number of parents/children aboard |
| `Fare` | Passenger fare |
| `Embarked` | Port of embarkation (C, Q, S) |
| `Cabin` | Cabin number |

---

## Projects

### 1. Decision Tree

**Notebook:** `decision-tree/Decision_Tree.ipynb`
**Spec:** `docs/Mini-projeto-arvoresdecisao2025.pdf`

Explores tree-based models for binary classification of Titanic survival.

**Preprocessing**
- Removal of low-signal features (`Name`, `Ticket`, `PassengerId`)
- Cabin grouped into binary: has cabin / no cabin
- Missing `Age` imputed using median per sex and class
- Missing `Embarked` filled with mode
- Categorical variables encoded via one-hot encoding
- Family size feature engineered from `SibSp + Parch + 1`
- Numerical features normalized with MinMaxScaler

**Modeling — Experiments**

| # | Model | Key Hyperparameters | Validation Accuracy |
|---|-------|---------------------|---------------------|
| 0 | Decision Tree | criterion=gini, max_depth=5, min_samples_split=10 | 78.92% |
| 1 | Decision Tree | criterion=entropy, max_depth=5, min_samples_split=10 | 81.17% |
| 2 | Decision Tree | criterion=entropy, max_depth=8, min_samples_split=20 | 80.72% |
| 3 | Decision Tree | criterion=entropy, max_depth=8, min_samples_split=10 | 82.96% |
| 4 | Decision Tree | criterion=entropy, max_depth=4, min_samples_split=10 | 79.37% |
| 5 | Random Forest | max_depth=8, min_samples_split=10, min_samples_leaf=4 | 82.51% |
| 6 | Gradient Boosting | max_depth=8, min_samples_split=20 | 86.10% |

**Hyperparameter Tuning:** Grid Search and Optuna were used to optimize Gradient Boosting parameters, achieving the best overall performance.

---

### 2. Multi-Layer Perceptron (MLP)

**Notebook:** `mlp/MLP.ipynb`
**Spec:** `docs/Mini-projeto-MLP2025.pdf`

Explores neural network architectures for binary classification of Titanic survival.

**Preprocessing**
- Same pipeline as Decision Tree (imputation, encoding, normalization)
- Features selected: `Pclass`, `Sex`, `Age`, `Fare`, `Embarked`, `Cabin`, `FamilySize`

**Modeling — Experiments**

| # | Architecture | Activation | Optimizer | Regularization | Validation Accuracy |
|---|-------------|------------|-----------|----------------|---------------------|
| 0 | Baseline (shallow) | ReLU | Adam | None | baseline |
| 1 | Deeper layers | ReLU | Adam | None | improved |
| 2 | Deeper layers | ReLU | Adam | Dropout | improved |
| 3 | Adjusted depth | Tanh | SGD | L2 | tested |
| 4 | Optimized depth | ReLU | Adam | Dropout + L2 | best manual |

**Hyperparameter Tuning:** Grid Search and Optuna were applied over layer sizes, learning rate, dropout rate, and optimizer type to find the best MLP configuration.

---

### 3. Unsupervised Clustering

**Notebook:** `unsupervised-clustering/Unsupervised.ipynb`
**Spec:** `docs/Mini-projeto-Kmeans-K-prototypes-DBSCAN.pdf`

Applies unsupervised learning algorithms to discover passenger groupings, then uses a Decision Tree to interpret the generated clusters.

**Preprocessing**
- Same base pipeline used across all projects
- Mixed data handled by K-Prototypes (numerical + categorical features simultaneously)

**Algorithms**

| Algorithm | Type | Key Parameters |
|-----------|------|----------------|
| K-Means | Partitional | n_clusters tuned via Elbow Method and Silhouette Score |
| K-Prototypes | Partitional (mixed data) | n_clusters, gamma (categorical weight) |
| DBSCAN | Density-based | eps and min_samples tuned via k-distance graph |

**Methodology**
1. **K-Means** — applied on numerical features; optimal k selected via Elbow Method and Silhouette Score
2. **K-Prototypes** — handles mixed (numerical + categorical) features natively, avoiding the need for full one-hot encoding
3. **DBSCAN** — density-based clustering; identifies noise points and arbitrary-shaped clusters without requiring a fixed k
4. **Cluster Interpretation** — a Decision Tree trained on cluster labels (from K-Means) to extract human-readable rules defining each group
5. **Survival Analysis** — average survival rate computed per cluster to validate whether groupings capture meaningful patterns

---

## Getting Started

### Prerequisites

- Python 3.8+
- Jupyter Notebook or Google Colab

### Install dependencies

```bash
pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn optuna kmodes
```

### Run a notebook

```bash
jupyter notebook decision-tree/Decision_Tree.ipynb
```

Or upload any `.ipynb` file directly to **Google Colab**.

---

## Tech Stack

- **Python 3**
- **Pandas** — data manipulation
- **NumPy** — numerical operations
- **Matplotlib / Seaborn** — visualization
- **Scikit-learn** — modeling, preprocessing, and evaluation
- **Imbalanced-learn** — class balancing
- **Optuna** — hyperparameter optimization
- **KModes** — K-Prototypes for mixed-type clustering

---

## License

Academic project — for educational use only. CIn/UFPE, 2025.
