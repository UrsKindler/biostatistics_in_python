### Overview
- **Purpose**: Comprehensively evaluate machine learning model performance across classification and regression tasks using clinically relevant performance metrics beyond simple accuracy.
- **Use Case**: Diagnostic biomarker validation, disease risk scoring, drug response classification, and regression-based survival/time-to-event estimation.
- **Prerequisites**: 
	* Python 3.9+
	* Ground truth labels ($y_{\text{true}}$) and model predictions ($y_{\text{pred}}$ / probability scores $y_{\text{prob}}$) on an independent test set
- **Data Types**: 
	* Categorical binary / multiclass labels (Classification)
	* Continuous numerical values (Regression)
- **Output**:
	* Confusion Matrix, Sensitivity/Recall, Specificity, Precision, F1-Score, ROC-AUC, Brier Score, MAE, RMSE, $R^2$
	* Dual-panel evaluation figure (`14_model_evaluation_metrics.png`) displaying Confusion Matrix and ROC Curve

#### When to Use

| Problem Type | Target Metric | Mathematical Definition | Clinical / Biological Relevance | Recommended Python Function |
| :--- | :--- | :--- | :--- | :--- |
| **Diagnostic Screening** | **Sensitivity (Recall)** | $\frac{TP}{TP + FN}$ | Minimizes False Negatives (identifies all sick patients) | `sklearn.metrics.recall_score` |
| **Confirmatory Testing** | **Precision (PPV)** | $\frac{TP}{TP + FP}$ | Minimizes False Positives (avoids unnecessary aggressive treatment) | `sklearn.metrics.precision_score` |
| **Unbalanced Classification** | **F1-Score / PR-AUC** | $2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$ | Harmonized trade-off on minority disease classes | `sklearn.metrics.f1_score` |
| **Threshold-Independent** | **ROC-AUC** | Area under $\text{TPR}$ vs. $\text{FPR}$ | Overall discrimination capability across all decision cut-offs | `sklearn.metrics.roc_auc_score` |
| **Continuous Regression** | **RMSE / $R^2$** | $\sqrt{\frac{1}{N}\sum (y_i - \hat{y}_i)^2}$ | Error magnitude in original physical/biological measurement units | `sklearn.metrics.root_mean_squared_error` |

#### Decision Criteria
- **Use ROC-AUC and F1-Score always**: For medical diagnostic datasets where positive disease cases are rare ($<15\%$).
- **Never rely on Accuracy alone**: In a dataset with 95% healthy and 5% sick patients, a dummy model predicting "healthy" for everyone achieves 95% accuracy while missing 100% of sick patients.
- **Don't skip when**: Comparing multiple candidate machine learning models for clinical deployment.

### Python Libraries & Methods

| Aspect | `sklearn.metrics` Classification | `sklearn.metrics` Regression |
| :--- | :--- | :--- |
| **Key Functions** | `classification_report`, `confusion_matrix`, `roc_curve`, `roc_auc_score` | `mean_absolute_error`, `root_mean_squared_error`, `r2_score` |
| **Input Arguments** | `(y_true, y_pred)` oder `(y_true, y_prob)` | `(y_true, y_pred)` |
| **Multi-Class Support** | `average='weighted'`, `'macro'`, `'micro'` | Built-in multi-output support |

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
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split


def generate_evaluation_data(seed: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Simulates an unbalanced clinical classification cohort."""
    X, y = make_classification(
        n_samples=300,
        n_features=10,
        n_informative=4,
        weights=[0.75, 0.25], # 25% Positives
        random_state=seed
    )
    return train_test_split(X, y, test_size=0.30, random_state=seed, stratify=y)


def main() -> None:
    X_train, X_test, y_train, y_test = generate_evaluation_data()
    
    # Modell trainieren
    rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    rf.fit(X_train, y_train)
    
    # Vorhersagen generieren
    y_pred = rf.predict(X_test)
    y_prob = rf.predict_proba(X_test)[:, 1]
    
    # 1. Metriken berechnen
    sens = recall_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)
    
    print("=== CLINICAL MODEL EVALUATION SUMMARY ===")
    print(f"Sensitivity (Recall): {sens:.3f}")
    print(f"Precision (PPV):       {prec:.3f}")
    print(f"F1-Score:             {f1:.3f}")
    print(f"ROC-AUC:              {auc:.3f}")
    
    print("\nFull Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Healthy (0)", "Diseased (1)"], digits=3))
    
    # 2. Visualisierung
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    
    # Panel A: Konfusionsmatrix Heatmap
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axes[0],
                xticklabels=["Pred: Healthy", "Pred: Diseased"],
                yticklabels=["True: Healthy", "True: Diseased"],
                annot_kws={"size": 14, "fontweight": "bold"})
    axes[0].set_title(f"A: Konfusionsmatrix (Test Set N={len(y_test)})\nF1-Score = {f1:.3f}", fontweight="bold", fontsize=11)
    
    # Panel B: ROC-Kurve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    axes[1].plot(fpr, tpr, color="#2C7FB8", lw=2.5, label=f"Random Forest (AUC = {auc:.3f})")
    axes[1].plot([0, 1], [0, 1], color="gray", linestyle="--", label="Zufalls-Klassifikator (AUC = 0.50)")
    axes[1].set_title("B: ROC-Kurve (Receiver Operating Characteristic)", fontweight="bold", fontsize=11)
    axes[1].set_xlabel("False Positive Rate (1 - Specificity)")
    axes[1].set_ylabel("True Positive Rate (Sensitivity / Recall)")
    axes[1].legend(loc="lower right")
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    
    output_file = Path("14_model_evaluation_metrics.png").resolve()
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    print(f"\nVisualisierung erfolgreich gespeichert unter:\n{output_file}")


if __name__ == "__main__":
    main()
```

output example:
![[14_model_evaluation_metrics.png]]
