### Overview
- **Purpose**: Resample and partition training datasets into multiple internal train/validation folds to reliably estimate model generalization performance, guard against severe overfitting, and tune hyperparameters without test set data leakage.
- **Use Case**: Evaluating biomarker classification models, optimizing tree depths / regularization parameters, and assessing model stability across small patient cohorts.
- **Prerequisites**: 
	* Python 3.9+
	* Feature matrix $X$ and target labels $y$
- **Data Types**: 
	* Continuous or discrete feature predictors
	* Categorical binary/multiclass target classes or continuous regression targets
- **Output**:
	* Fold-by-fold validation performance scores, mean accuracy, and standard error
	* Comparative 2-panel CV performance and fold variability figure (`13_cross_validation.png`)

#### When to Use

| Cross-Validation Strategy | Data Scenario | Class Balance | Recommended Python Class | Note |
| :--- | :--- | :--- | :--- | :--- |
| **Stratified K-Fold** | Classification with unbalanced classes | Unbalanced (e.g. 10% diseased) | `sklearn.model_selection.StratifiedKFold` | **Default choice for biological classification**; preserves class ratios in every fold |
| **Standard K-Fold** | Continuous regression or balanced classes | Balanced | `sklearn.model_selection.KFold` | Shuffles and splits data into $k$ equal chunks |
| **Leave-One-Out (LOOCV)** | Very small biological cohorts ($N < 30$) | Any | `sklearn.model_selection.LeaveOneOut` | $k = N$; maximum training data per fold, but high computational cost and variance |
| **Repeated Stratified K-Fold** | High-variance omics datasets | Unbalanced | `sklearn.model_selection.RepeatedStratifiedKFold` | Runs $n$-repeats of $k$-fold splits to stabilize performance estimates |

#### Decision Criteria
- **Use Stratified K-Fold always**: For clinical diagnostic / biomarker classification where diseased vs. healthy sample counts are unequal.
- **Essential choice of $k$**: $k=5$ or $k=10$ provides the best empirical balance between bias and variance.
- **Critical rule**: Any preprocessing (imputation, scaling, feature selection) must be fitted **inside** the CV loop (e.g., using `sklearn.pipeline.Pipeline`) to avoid data leakage!

### Python Libraries & Methods

| Aspect | `StratifiedKFold` | `KFold` | `cross_val_score` |
| :--- | :--- | :--- | :--- |
| **Module** | `sklearn.model_selection` | `sklearn.model_selection` | `sklearn.model_selection` |
| **Stratification** | Yes (preserves $y$ proportions) | No (purely random split) | Delegates to CV splitter |
| **Key Arguments** | `n_splits=5`, `shuffle=True`, `random_state=42` | `n_splits=5`, `shuffle=True` | `estimator`, `X`, `y`, `cv`, `scoring='roc_auc'` |
| **Returns** | Generator of `(train_idx, test_idx)` | Generator of `(train_idx, test_idx)` | 1D Array of fold scores |

### Quick Start Code

```bash
python -m pip install scikit-learn numpy pandas matplotlib seaborn
```

```python
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, RepeatedStratifiedKFold, StratifiedKFold, cross_val_score


def generate_cohort_cv_data(n_samples: int = 150, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Generates synthetic unbalanced biomedical dataset (80% Class 0, 20% Class 1)."""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=12,
        n_informative=5,
        weights=[0.80, 0.20],
        random_state=seed
    )
    return X, y


def main() -> None:
    X, y = generate_cohort_cv_data()
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    
    # 1. Stratified 5-Fold CV (Empfohlen für unbalancierte Biomarker-Daten)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores_skf = cross_val_score(model, X, y, cv=skf, scoring="roc_auc")
    
    # 2. Standard 5-Fold CV (Zum Vergleich)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores_kf = cross_val_score(model, X, y, cv=kf, scoring="roc_auc")
    
    # 3. Repeated Stratified K-Fold (5 Splits x 5 Repeats)
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)
    scores_rskf = cross_val_score(model, X, y, cv=rskf, scoring="roc_auc")
    
    print("=== CROSS-VALIDATION PERFORMANCE SUMMARY (ROC-AUC) ===")
    print(f"Stratified 5-Fold: {scores_skf.mean():.3f} (+/- {scores_skf.std():.3f}) | Folds: {scores_skf.round(3)}")
    print(f"Standard 5-Fold:   {scores_kf.mean():.3f} (+/- {scores_kf.std():.3f}) | Folds: {scores_kf.round(3)}")
    print(f"Repeated 5x5 Fold: {scores_rskf.mean():.3f} (+/- {scores_rskf.std():.3f}) (25 Iterationen)")
    
    # 4. Visualisierung
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Panel A: Fold-by-Fold Performance Vergleich
    fold_df = pd.DataFrame({
        "Fold": [f"Fold {i+1}" for i in range(5)],
        "Stratified K-Fold": scores_skf,
        "Standard K-Fold": scores_kf
    }).melt(id_vars="Fold", var_name="CV Strategy", value_name="ROC_AUC")
    
    sns.barplot(data=fold_df, x="Fold", y="ROC_AUC", hue="CV Strategy", palette=["#2C7FB8", "#E6550D"], ax=axes[0])
    axes[0].axhline(scores_skf.mean(), color="#2C7FB8", linestyle="--", alpha=0.7, label="Stratified Mean")
    axes[0].set_title("A: Fold-Stabilität (ROC-AUC pro Split)", fontweight="bold", fontsize=11)
    axes[0].set_ylim(0.5, 1.0)
    axes[0].legend(loc="lower right")
    axes[0].grid(axis="y", alpha=0.3)
    
    # Panel B: Verteilung der Scores (Repeated CV)
    sns.boxplot(y=scores_rskf, ax=axes[1], color="#41B6C4", width=0.35)
    sns.stripplot(y=scores_rskf, ax=axes[1], color="black", alpha=0.7, jitter=0.15, s=6)
    axes[1].set_title(f"B: Repeated Stratified CV (25 Folds)\nMean AUC = {scores_rskf.mean():.3f} (+/- {scores_rskf.std():.3f})",
                      fontweight="bold", fontsize=11)
    axes[1].set_ylabel("ROC-AUC Score")
    axes[1].set_ylim(0.5, 1.0)
    axes[1].grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    
    output_file = Path("13_cross_validation.png").resolve()
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    print(f"\nVisualisierung erfolgreich gespeichert unter:\n{output_file}")


if __name__ == "__main__":
    main()
```

output example:
![[13_cross_validation.png]]
