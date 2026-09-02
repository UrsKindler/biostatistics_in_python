from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree


def plot_tree_and_nn_decision(X: np.ndarray, y: np.ndarray, outpath: Path) -> None:
    dt = DecisionTreeClassifier(max_depth=3, random_state=42)
    dt.fit(X, y)

    mlp = MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=500, random_state=42)
    mlp.fit(X, y)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Panel A: Decision Tree Flowchart
    plot_tree(dt, filled=True, feature_names=[f"Protein_{i+1}" for i in range(X.shape[1])], class_names=["Control", "Disease"], ax=axes[0])
    axes[0].set_title("A: Interpretable Decision Tree Architecture", fontweight="bold", fontsize=11)

    # Panel B: MLP Loss Curve
    axes[1].plot(mlp.loss_curve_, color="#2C7FB8", lw=2)
    axes[1].set_title("B: Neural Network (MLP) Optimization Loss Curve", fontweight="bold", fontsize=11)
    axes[1].set_xlabel("Training Iterations (Epochs)")
    axes[1].set_ylabel("Cross-Entropy Loss")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    outpath.parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    np.random.seed(42)
    X = np.random.normal(10, 2, (80, 10))
    y = np.where(X[:, 0] + X[:, 1] > 20, 1, 0)

    out_dir = Path(__file__).parent
    outpath = out_dir / "12_decision_trees_and_neural_nets.png"
    plot_tree_and_nn_decision(X, y, outpath)
    print(f"Decision tree and NN plot saved to: {outpath.resolve()}")


if __name__ == "__main__":
    main()
