from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import interpolate, stats
from statsmodels.stats.multitest import multipletests


def estimate_pi0(p_values: np.ndarray, lambdas: np.ndarray | None = None) -> float:
    p = np.asarray(p_values)
    p = p[~np.isnan(p)]
    m = len(p)
    if m == 0:
        return 1.0

    if lambdas is None:
        lambdas = np.linspace(0.05, 0.90, 18)

    pi0_lambda = np.array([np.sum(p > l) / (m * (1.0 - l)) for l in lambdas])

    try:
        spline = interpolate.UnivariateSpline(lambdas, pi0_lambda, k=3, s=None)
        pi0_est = float(spline(1.0))
    except Exception:
        pi0_est = float(np.mean(pi0_lambda[-4:]))

    return float(np.clip(pi0_est, 0.0, 1.0))


def calculate_storey_qvalues(p_values: np.ndarray, pi0: float | None = None) -> np.ndarray:
    p = np.asarray(p_values)
    valid_mask = ~np.isnan(p)
    p_valid = p[valid_mask]
    m = len(p_valid)
    if m == 0:
        return p.copy()

    if pi0 is None:
        pi0 = estimate_pi0(p_valid)

    order = np.argsort(p_valid)
    ranked_p = p_valid[order]
    ranks = np.arange(1, m + 1)

    q_raw = (pi0 * m * ranked_p) / ranks
    q_raw = np.clip(q_raw, 0.0, 1.0)
    q_mono = np.minimum.accumulate(q_raw[::-1])[::-1]

    q_values = np.empty_like(p_valid)
    q_values[order] = q_mono

    out = np.full_like(p, np.nan)
    out[valid_mask] = q_values
    return out


def run_multiple_testing_correction(
    p_values: np.ndarray,
    alpha: float = 0.05,
) -> pd.DataFrame:
    df_res = pd.DataFrame({"p_raw": p_values})

    df_res["p_bonferroni"] = multipletests(p_values, alpha=alpha, method="bonferroni")[1]
    df_res["p_holm"] = multipletests(p_values, alpha=alpha, method="holm")[1]
    df_res["p_hochberg"] = multipletests(p_values, alpha=alpha, method="simes-hochberg")[1]
    df_res["fdr_bh"] = multipletests(p_values, alpha=alpha, method="fdr_bh")[1]
    df_res["fdr_by"] = multipletests(p_values, alpha=alpha, method="fdr_by")[1]

    pi0 = estimate_pi0(p_values)
    df_res["q_storey"] = calculate_storey_qvalues(p_values, pi0=pi0)

    return df_res


def plot_multiple_correction_benchmark(
    df_res: pd.DataFrame,
    alpha: float = 0.05,
    outpath: Path = Path("09_3_false_discovery_rate.png"),
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    p_vals = df_res["p_raw"].dropna()
    pi0 = estimate_pi0(p_vals.values)
    axes[0, 0].hist(p_vals, bins=40, density=True, color="#4C72B0", alpha=0.75, edgecolor="black")
    axes[0, 0].axhline(pi0, color="crimson", linestyle="--", lw=2, label=f"Estimated $\hat{{\pi}}_0 = {pi0:.3f}$")
    axes[0, 0].set_xlabel("Raw P-value")
    axes[0, 0].set_ylabel("Density")
    axes[0, 0].set_title("A: P-value Histogram & Storey $\pi_0$ Baseline", fontweight="bold", fontsize=11)
    axes[0, 0].legend(loc="upper right", frameon=True)
    axes[0, 0].grid(alpha=0.3)

    thresholds = np.linspace(0.001, 0.20, 100)
    methods = [
        ("p_raw", "Raw P (Uncorrected)", "gray", ":"),
        ("fdr_bh", "Benjamini-Hochberg (BH)", "#2C7FB8", "-"),
        ("q_storey", "Storey q-value (pFDR)", "#5CB85C", "--"),
        ("fdr_by", "Benjamini-Yekutieli (BY)", "#E6550D", "-."),
        ("p_holm", "Holm (FWER)", "#756BB1", "-"),
        ("p_bonferroni", "Bonferroni (FWER)", "black", "--"),
    ]

    for col, label, color, ls in methods:
        rejections = [np.sum(df_res[col] <= t) for t in thresholds]
        axes[0, 1].plot(thresholds, rejections, label=label, color=color, linestyle=ls, lw=1.8)

    axes[0, 1].axvline(alpha, color="red", linestyle=":", alpha=0.8, label=f"$\\alpha = {alpha}$")
    axes[0, 1].set_xlabel("Significance Threshold ($\\alpha$ / FDR cut-off)")
    axes[0, 1].set_ylabel("Number of Significant Discoveries")
    axes[0, 1].set_title("B: Power & Discoveries vs Error Threshold", fontweight="bold", fontsize=11)
    axes[0, 1].legend(loc="upper left", fontsize=8, frameon=True)
    axes[0, 1].grid(alpha=0.3)

    p_sort = df_res.sort_values("p_raw")
    axes[1, 0].scatter(p_sort["p_raw"], p_sort["fdr_bh"], s=8, color="#2C7FB8", label="BH FDR", alpha=0.7)
    axes[1, 0].scatter(p_sort["p_raw"], p_sort["q_storey"], s=8, color="#5CB85C", label="Storey q-value", alpha=0.7)
    axes[1, 0].scatter(p_sort["p_raw"], p_sort["p_holm"], s=8, color="#756BB1", label="Holm FWER", alpha=0.7)
    axes[1, 0].plot([0, 0.1], [0, 0.1], "k--", lw=1, label="Identity line")
    axes[1, 0].set_xlim(0, 0.08)
    axes[1, 0].set_ylim(0, 0.25)
    axes[1, 0].set_xlabel("Raw P-value")
    axes[1, 0].set_ylabel("Adjusted Value / FDR")
    axes[1, 0].set_title("C: Adjusted Significance Calibration (Zoom)", fontweight="bold", fontsize=11)
    axes[1, 0].legend(loc="upper left", fontsize=8, frameon=True)
    axes[1, 0].grid(alpha=0.3)

    counts = {
        "Raw P": (df_res["p_raw"] <= alpha).sum(),
        "Storey q": (df_res["q_storey"] <= alpha).sum(),
        "BH FDR": (df_res["fdr_bh"] <= alpha).sum(),
        "BY FDR": (df_res["fdr_by"] <= alpha).sum(),
        "Holm": (df_res["p_holm"] <= alpha).sum(),
        "Bonferroni": (df_res["p_bonferroni"] <= alpha).sum(),
    }
    bars = axes[1, 1].bar(counts.keys(), counts.values(), color=["gray", "#5CB85C", "#2C7FB8", "#E6550D", "#756BB1", "black"], alpha=0.85)
    for bar in bars:
        yval = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2.0, yval + 2, f"{int(yval)}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    axes[1, 1].set_ylabel(f"Significant Features (at $\\alpha={alpha}$)")
    axes[1, 1].set_title(f"D: Total Discoveries at Nominal $\\alpha = {alpha}$", fontweight="bold", fontsize=11)
    axes[1, 1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    outpath.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()


def main() -> None:
    out_dir = Path(__file__).parent
    np.random.seed(42)
    m = 2000
    m0 = int(0.85 * m)
    m1 = m - m0

    p_null = np.random.uniform(0, 1, m0)
    p_alt = np.random.beta(0.3, 4, m1)
    p_all = np.concatenate([p_null, p_alt])

    df_res = run_multiple_testing_correction(p_all, alpha=0.05)
    out_path = out_dir / "09_3_false_discovery_rate.png"
    plot_multiple_correction_benchmark(df_res, alpha=0.05, outpath=out_path)
    print(f"Multiple correction benchmark saved to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
