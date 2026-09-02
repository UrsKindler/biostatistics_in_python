from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, LeaveOneOut, StratifiedKFold


def plot_cv_schemes(n_samples: int, y: np.ndarray, outpath: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # K-Fold
    min_class = np.min(np.bincount(y)) if len(np.unique(y)) > 1 and len(y) > 0 else n_samples
    k_splits = min(5, min_class) if min_class >= 2 else 2

    kf = KFold(n_splits=k_splits)
    for fold, (_, val_idx) in enumerate(kf.split(range(n_samples))):
        axes[0].scatter(val_idx, [fold] * len(val_idx), marker="_", lw=6, color="#2C7FB8")
    axes[0].set_title(f"A: Standard {k_splits}-Fold CV", fontweight="bold")
    axes[0].set_xlabel("Sample Index")
    axes[0].set_ylabel("Fold Index")

    # Stratified K-Fold
    skf = StratifiedKFold(n_splits=k_splits)
    for fold, (_, val_idx) in enumerate(skf.split(range(n_samples), y)):
        colors = ["#2C7FB8" if y[i] == 0 else "#D9534F" for i in val_idx]
        axes[1].scatter(val_idx, [fold] * len(val_idx), c=colors, marker="_", lw=6)
    axes[1].set_title(f"B: Stratified {k_splits}-Fold CV (Balanced Classes)", fontweight="bold")
    axes[1].set_xlabel("Sample Index")

    # LOOCV
    loo = LeaveOneOut()
    for fold, (_, val_idx) in enumerate(loo.split(range(n_samples))):
        if fold < 15:
            axes[2].scatter(val_idx, [fold] * len(val_idx), marker="_", lw=6, color="#5CB85C")
    axes[2].set_title("C: Leave-One-Out (First 15 Folds)", fontweight="bold")
    axes[2].set_xlabel("Sample Index")

    plt.tight_layout()
    outpath.parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    n_samples = 40
    y = np.array([0] * 20 + [1] * 20)
    out_dir = Path(__file__).parent
    outpath = out_dir / "13_cross_validation.png"
    plot_cv_schemes(n_samples, y, outpath)
    print(f"Cross-validation plot saved to: {outpath.resolve()}")


if __name__ == "__main__":
    main()
