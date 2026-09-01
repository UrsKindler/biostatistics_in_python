### Overview
- **Purpose**: Train supervised machine learning models to classify biological phenotypes or predict continuous clinical outcomes from multi-omics feature matrices, comparing interpretable rule-based trees with flexible non-linear Multi-Layer Perceptron (MLP) neural networks.
- **Use Case**: Diagnostic disease classification (Responder vs. Non-Responder), patient survival risk stratification, or non-linear multi-biomarker scoring.
- **Prerequisites**: 
	* Python 3.9+
	* Labeled training and test datasets ($X_{\text{train}}, y_{\text{train}}, X_{\text{test}}, y_{\text{test}}$)
- **Data Types**: 
	* Continuous numerical features (standardized for Neural Networks)
	* Categorical / discrete feature columns (Decision Trees)
	* Binary or multi-class target labels ($y$)
- **Output**:
	* Accuracy, ROC-AUC, Classification Report, Feature Importances (Trees)
	* 2-Panel model performance and feature importance plot (`12_decision_trees_and_neural_nets.png`)

#### When to Use

| Model Family | Algorithm | Strengths | Weaknesses | Recommended Python Class |
| :--- | :--- | :--- | :--- | :--- |
| **Decision Trees** | CART (Classification and Regression Trees) | Highly interpretable, handles non-linear splits natively, provides exact feature decision thresholds | Prone to overfitting / high variance if depth is unrestricted | `sklearn.tree.DecisionTreeClassifier(max_depth=4)` |
| **Neural Networks** | Multi-Layer Perceptron (MLP) | Learns complex high-order non-linear interaction surfaces | "Black-box" model; requires careful feature scaling and hyperparameter regularization | `sklearn.neural_network.MLPClassifier(hidden_layer_sizes=(32, 16))` |

#### Decision Criteria
- **Use Decision Trees when**: Clinical interpretability, transparent decision rules, and biomarker threshold values are required by regulatory bodies or clinicians.
- **Use Neural Networks (MLP) when**: Sample size is sufficiently large, features are continuous/scaled, and biological interactions are highly non-linear.
- **Critical rule**: Always train models strictly on `X_train` and evaluate on a held-out `X_test` to prevent data leakage and over-optimistic performance estimates.

### Python Libraries & Methods

| Aspect | `sklearn.tree.DecisionTreeClassifier` | `sklearn.neural_network.MLPClassifier` |
| :--- | :--- | :--- |
| **Interpretability** | White-box (rule extraction, `export_text`) | Black-box (dense weight matrices) |
| **Scaling Dependency** | Invariant to monotonic feature scaling | Strictly requires standard scaling (`StandardScaler`) |
| **Key Hyperparameters** | `max_depth`, `min_samples_split`, `criterion` | `hidden_layer_sizes`, `alpha` (L2 penalty), `max_iter` |
| **Feature Importance** | `tree.feature_importances_` (Gini Impurity) | Permutation importance / SHAP values |

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
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


def generate_biomarker_cohort(n_samples: int = 250, n_features: int = 15, seed: int = 42) -> tuple[pd.DataFrame, pd.Series]:
    """Generates synthetic patient classification cohort with informative and noise features."""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=6,
        n_redundant=2,
        n_classes=2,
        weights=[0.6, 0.4],
        random_state=seed
    )
    feature_names = [f"Biomarker_{i+1:02d}" for i in range(n_features)]
    return pd.DataFrame(X, columns=feature_names), pd.Series(y, name="Treatment_Response")


def main() -> None:
    X, y = generate_biomarker_cohort()
    
    # 1. Train/Test Split (70/30)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
    
    # 2. Skalierung für neuronales Netz
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 3. Modelle trainieren
    # A: Entscheidungsbaum (begrenzte Tiefe gegen Overfitting)
    dt = DecisionTreeClassifier(max_depth=4, min_samples_split=6, random_state=42)
    dt.fit(X_train, y_train)
    dt_pred = dt.predict(X_test)
    dt_prob = dt.predict_proba(X_test)[:, 1]
    
    # B: Multi-Layer Perceptron (MLP)
    mlp = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=800, alpha=0.01, random_state=42)
    mlp.fit(X_train_scaled, y_train)
    mlp_pred = mlp.predict(X_test_scaled)
    mlp_prob = mlp.predict_proba(X_test_scaled)[:, 1]
    
    # 4. Leistungsmetriken ausgeben
    print("=== SUPERVISED MODEL EVALUATION (TEST SET) ===")
    print(f"Decision Tree Accuracy: {accuracy_score(y_test, dt_pred):.3f} | ROC-AUC: {roc_auc_score(y_test, dt_prob):.3f}")
    print(f"Neural Net (MLP) Acc:   {accuracy_score(y_test, mlp_pred):.3f} | ROC-AUC: {roc_auc_score(y_test, mlp_prob):.3f}")
    
    print("\nDecision Tree Classification Report:")
    print(classification_report(y_test, dt_pred, digits=3))
    
    # 5. Visualisierung
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Panel A: Feature Importances des Decision Trees
    importances = pd.Series(dt.feature_importances_, index=X.columns).sort_values(ascending=True)
    importances.tail(8).plot(kind="barh", color="#2C7FB8", edgecolor="black", ax=axes[0])
    axes[0].set_title("A: Decision Tree Feature Importance (Gini)", fontweight="bold", fontsize=11)
    axes[0].set_xlabel("Relative Wichtigkeit")
    axes[0].grid(axis="x", alpha=0.3)
    
    # Panel B: Vorhergesagte Wahrscheinlichkeiten (MLP)
    df_prob = pd.DataFrame({"True_Label": y_test.map({0: "Non-Responder", 1: "Responder"}), "Predicted_Prob": mlp_prob})
    sns.boxplot(data=df_prob, x="True_Label", y="Predicted_Prob", palette="Set2", width=0.45, ax=axes[1])
    sns.stripplot(data=df_prob, x="True_Label", y="Predicted_Prob", color="black", alpha=0.6, jitter=0.2, ax=axes[1])
    axes[1].axhline(0.5, color="red", linestyle="--", label="Klassifikationsgrenze (0.5)")
    axes[1].set_title("B: MLP Vorhersage-Wahrscheinlichkeiten (Test Set)", fontweight="bold", fontsize=11)
    axes[1].set_ylabel("Modellwahrscheinlichkeit (P(Response))")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    
    output_file = Path("12_decision_trees_and_neural_nets.png").resolve()
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    print(f"\nVisualisierung erfolgreich gespeichert unter:\n{output_file}")


if __name__ == "__main__":
    main()
```

output example:
![[12_decision_trees_and_neural_nets.png]]
