from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import auc, confusion_matrix, precision_recall_curve, roc_curve


def plot_model_metrics(y_true: np.ndarray, y_score: np.ndarray, outpath: Path) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    precision, recall, _ = precision_recall_curve(y_true, y_score)
    pr_auc = auc(recall, precision)

    y_pred = (y_score >= 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # A: ROC Curve
    axes[0].plot(fpr, tpr, color="#2C7FB8", lw=2.5, label=f"ROC (AUC = {roc_auc:.3f})")
    axes[0].plot([0, 1], [0, 1], "k--", lw=1)
    axes[0].set_title("A: Receiver Operating Characteristic (ROC)", fontweight="bold")
    axes[0].set_xlabel("False Positive Rate (1 - Specificity)")
    axes[0].set_ylabel("True Positive Rate (Sensitivity)")
    axes[0].legend(loc="lower right")
    axes[0].grid(alpha=0.3)

    # B: PR Curve
    axes[1].plot(recall, precision, color="#5CB85C", lw=2.5, label=f"PR (AUC = {pr_auc:.3f})")
    axes[1].set_title("B: Precision-Recall Curve", fontweight="bold")
    axes[1].set_xlabel("Recall (Sensitivity)")
    axes[1].set_ylabel("Precision (PPV)")
    axes[1].legend(loc="lower left")
    axes[1].grid(alpha=0.3)

    # C: Confusion Matrix
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axes[2], annot_kws={"size": 14, "weight": "bold"})
    axes[2].set_title("C: Confusion Matrix (Cutoff = 0.5)", fontweight="bold")
    axes[2].set_xlabel("Predicted Class")
    axes[2].set_ylabel("True Class")

    plt.tight_layout()
    outpath.parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    np.random.seed(42)
    y_true = np.array([0] * 50 + [1] * 50)
    y_score = np.concatenate([np.random.beta(2, 5, 50), np.random.beta(5, 2, 50)])

    out_dir = Path(__file__).parent
    outpath = out_dir / "14_model_evaluation_metrics.png"
    plot_model_metrics(y_true, y_score, outpath)
    print(f"Model evaluation metrics saved to: {outpath.resolve()}")


if __name__ == "__main__":
    main()
