from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance


def plot_rf_feature_importance(rf_model: RandomForestClassifier, X: np.ndarray, y: np.ndarray, feature_names: list[str], outpath: Path) -> None:
    gini_imp = rf_model.feature_importances_
    perm_res = permutation_importance(rf_model, X, y, n_repeats=10, random_state=42)
    perm_imp = perm_res.importances_mean

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    top_idx = np.argsort(gini_imp)[::-1][:10]

    axes[0].barh(np.array(feature_names)[top_idx][::-1], gini_imp[top_idx][::-1], color="#2C7FB8", alpha=0.85)
    axes[0].set_title("A: Gini Impurity Feature Importance", fontweight="bold", fontsize=11)
    axes[0].set_xlabel("Mean Decrease in Impurity (Gini)")
    axes[0].grid(axis="x", alpha=0.3)

    axes[1].barh(np.array(feature_names)[top_idx][::-1], perm_imp[top_idx][::-1], color="#5CB85C", alpha=0.85)
    axes[1].set_title("B: Permutation Feature Importance", fontweight="bold", fontsize=11)
    axes[1].set_xlabel("Mean Accuracy Drop upon Permutation")
    axes[1].grid(axis="x", alpha=0.3)

    plt.tight_layout()
    outpath.parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    np.random.seed(42)
    X = np.random.normal(10, 2, (80, 20))
    y = np.where(X[:, 2] * 2 + X[:, 5] > 25, 1, 0)
    feat_names = [f"Protein_{i+1:02d}" for i in range(20)]

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)

    out_dir = Path(__file__).parent
    outpath = out_dir / "15_random_forests.png"
    plot_rf_feature_importance(rf, X, y, feat_names, outpath)
    print(f"Random forest plot saved to: {outpath.resolve()}")


if __name__ == "__main__":
    main()
