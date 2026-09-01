### Overview
- **Purpose**: Train ensemble learning models aggregating hundreds of randomized decision trees (Bagging + Feature Subspace sampling) for high-accuracy biological phenotype classification, regression, and stable biomarker importance ranking.
- **Use Case**: High-dimensional multi-omics biomarker discovery (Proteomics, Transcriptomics), disease classification with complex non-linear feature interactions, and clinical outcome scoring.
- **Prerequisites**: 
	* Python 3.9+
	* Feature matrix $X$ and discrete labels $y$ (or continuous target values for regression)
- **Data Types**: 
	* Mixed numerical and encoded categorical features
	* Robust to moderate missingness and unscaled features
- **Output**:
	* Out-of-Bag (OOB) accuracy score, test set classification performance, and feature importance table
	* 2-Panel model performance and biomarker importance figure (`15_random_forests.png`)

#### When to Use

| Problem Scenario | Recommended Random Forest Class | Key Parameters | Note |
| :--- | :--- | :--- | :--- |
| **Omics Classification ($P \gg N$)** | `RandomForestClassifier` | `n_estimators=500`, `max_features='sqrt'`, `oob_score=True` | Highly resistant to overfitting; provides built-in OOB validation |
| **Continuous Clinical Outcomes** | `RandomForestRegressor` | `n_estimators=500`, `max_features=1.0/3.0`, `criterion='squared_error'` | Averages continuous predictions across tree ensembles |
| **High Feature Noise** | Random Forest with MDI & Permutation Importance | `n_jobs=-1`, `min_samples_leaf=2` | Aggregates feature importance across hundreds of randomized sub-trees |

#### Decision Criteria
- **Use Random Forest over Single Decision Tree**: Whenever prediction accuracy, generalization, and stability against noisy biological outliers are paramount.
- **Use Random Forest over Neural Networks**: When sample sizes are modest ($N < 500$) and transparent feature importance rankings are required for experimental validation.
- **Critical check**: Inspect `oob_score_` as an internal cross-validation proxy that uses samples not included in bootstrap training iterations.

### Python Libraries & Methods

| Aspect | `sklearn.ensemble.RandomForestClassifier` | `sklearn.ensemble.RandomForestRegressor` |
| :--- | :--- | :--- |
| **Task Type** | Discrete classification (Binary / Multi-class) | Continuous numeric regression |
| **Aggregation** | Majority voting / Average class probabilities | Arithmetic mean of individual tree outputs |
| **Internal Validation** | `oob_score=True` $\implies$ `model.oob_score_` | `oob_score=True` $\implies$ `model.oob_score_` ($R^2$) |
| **Feature Ranking** | `model.feature_importances_` (Mean Decrease in Impurity) | `model.feature_importances_` (Variance Reduction) |

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
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split


def generate_omics_biomarker_data(n_samples: int = 180, n_features: int = 40, seed: int = 42) -> tuple[pd.DataFrame, pd.Series]:
    """Generates high-dimensional simulated biomarker matrix with strong key drivers and background noise."""
    X_mat, y_vec = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=8,
        n_redundant=4,
        weights=[0.60, 0.40],
        random_state=seed
    )
    feature_names = [f"Protein_{i+1:02d}" for i in range(n_features)]
    return pd.DataFrame(X_mat, columns=feature_names), pd.Series(y_vec, name="Disease_Status")


def main() -> None:
    X, y = generate_omics_biomarker_data()
    
    # 1. Train / Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    
    # 2. Random Forest Classifier trainieren
    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=6,
        min_samples_split=4,
        max_features="sqrt",
        oob_score=True,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    
    # 3. Vorhersagen & Validierung
    y_pred = rf.predict(X_test)
    y_prob = rf.predict_proba(X_test)[:, 1]
    
    oob_acc = rf.oob_score_
    test_auc = roc_auc_score(y_test, y_prob)
    
    print("=== RANDOM FOREST CLASSIFICATION SUMMARY ===")
    print(f"Out-of-Bag (OOB) Accuracy: {oob_acc:.3f}")
    print(f"Test Set ROC-AUC:          {test_auc:.3f}")
    
    # 4. Feature Importances (Top 10 Biomarker)
    feat_imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("\nTop 10 Wichtigste Biomarker (MDI):")
    print(feat_imp.head(10).round(4))
    
    # 5. Visualisierung
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Panel A: Top 10 Feature Importances
    feat_imp.head(10).sort_values(ascending=True).plot(kind="barh", color="#2C7FB8", edgecolor="black", ax=axes[0])
    axes[0].set_title("A: Top 10 Biomarker Feature Importance (MDI Gini)", fontweight="bold", fontsize=11)
    axes[0].set_xlabel("Relative Wichtigkeit")
    axes[0].grid(axis="x", alpha=0.3)
    
    # Panel B: Verteilung der Vorhersagewahrscheinlichkeiten nach wahrer Klasse
    df_plot = pd.DataFrame({"Wahre Klasse": y_test.map({0: "Gesund", 1: "Erkrankt"}), "P(Erkrankt)": y_prob})
    sns.boxplot(data=df_plot, x="Wahre Klasse", y="P(Erkrankt)", palette=["#66C2A5", "#FC8D62"], width=0.4, ax=axes[1])
    sns.stripplot(data=df_plot, x="Wahre Klasse", y="P(Erkrankt)", color="black", alpha=0.7, jitter=0.2, s=7, ax=axes[1])
    axes[1].axhline(0.5, color="red", linestyle="--", label="Klassifikationsgrenze (0.5)")
    axes[1].set_title(f"B: Test Set Wahrscheinlichkeits-Trennung (AUC = {test_auc:.3f})\nOOB Accuracy = {oob_acc:.1%}",
                      fontweight="bold", fontsize=11)
    axes[1].set_ylabel("Modell-Wahrscheinlichkeit P(Erkrankt)")
    axes[1].legend(loc="upper left")
    axes[1].grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    
    output_file = Path("15_random_forests.png").resolve()
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    print(f"\nVisualisierung erfolgreich gespeichert unter:\n{output_file}")


if __name__ == "__main__":
    main()
```

output example:
![[15_random_forests.png]]
