"""
01b. Pairwise Scatter by Group (proteomics template)
========================================================
Purpose: For each group's replicates, plot every pairwise combination of
log2(Quantity) values against each other, annotate the coefficient of
determination (R^2), and flag pairs that deviate from perfect
reproducibility (y = x).

Use Case: Replicate-quality control in proteomics/omics runs -- quickly
spot a failed or batch-shifted replicate within a group before pooling
replicates for downstream statistics.

Prerequisites: pandas, numpy, scipy, matplotlib
Input: The shared random template CSV produced by
shared_random_proteomics_data.py (output/proteomics_template.csv). This
script reads that persisted file -- it never generates its own random
data, so results stay comparable with 01a, 02, etc.

Data Types: Per-run <run>.raw.PG.Quantity columns, grouped into
biological/experimental groups via GROUP_TO_RUNS.

Output: One PNG per call, one row per group, up to 3 pairwise scatter
panels per row (C(n_replicates, 2) pairs), each annotated with R^2 and a
red dashed y = x reference line.

This mirrors the make_pairwise_scatter_by_group() logic from the
original project script (drop_duplicate_columns, build_group_column_dict,
log_qly_group_summary, log2_transform_qty), adapted to the shared
proteomics template and its GROUP_TO_RUNS / COLOR_MAP definitions.
"""

import logging
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

from Random_Proteomics_Dataset_Generator import (
    TEMPLATE_PATH,
    GROUP_TO_RUNS,
    COLOR_MAP,
    load_template,
    create_random_proteomics_table,
    save_template,
)

GENERATOR_DIR = Path(__file__).resolve().parents[1] / "Random Proteomics Dataset Generator"
sys.path.insert(0, str(GENERATOR_DIR))

from Random_Proteomics_Dataset_Generator import (
    TEMPLATE_PATH, GROUP_TO_RUNS, load_template,
    create_random_proteomics_table, save_template,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("pairwise_scatter")


def drop_duplicate_columns(df: pd.DataFrame, context: str) -> pd.DataFrame:
    """Drop duplicated column names (keep first occurrence), logging how
    many were removed."""
    dup_mask = df.columns.duplicated()
    n_dupes = int(dup_mask.sum())
    if n_dupes:
        log.warning("%s: %d doppelte Spalten entfernt", context, n_dupes)
    return df.loc[:, ~dup_mask]


def build_group_column_dict(
    group_to_runs: dict[str, list[str]], available_cols: list[str]
) -> dict[str, list[str]]:
    """Map each group to its available '<run>.raw.PG.Quantity' columns,
    restricted to columns actually present in the DataFrame."""
    groups: dict[str, list[str]] = {}
    for group_name, runs in group_to_runs.items():
        qty_cols = [f"{run}.raw.PG.Quantity" for run in runs]
        valid_cols = [c for c in qty_cols if c in available_cols]
        if valid_cols:
            groups[group_name] = valid_cols
    return groups


def log_qly_group_summary(
    suffix: str,
    group_to_runs: dict[str, list[str]],
    groups_this_batch: dict[str, list[str]],
    available_cols: list[str],
) -> None:
    """Log how many Quantity columns per group were found vs. expected."""
    for group_name, runs in group_to_runs.items():
        expected = [f"{run}.raw.PG.Quantity" for run in runs]
        found = groups_this_batch.get(group_name, [])
        missing = [c for c in expected if c not in available_cols]
        log.info(
            "%s | Gruppe %s | erwartet=%d gefunden=%d fehlend=%d",
            suffix, group_name, len(expected), len(found), len(missing)
        )
        if missing:
            log.warning("%s | Gruppe %s | fehlende Spalten: %s", suffix, group_name, missing)


def log2_transform_qty(
    df: pd.DataFrame, qty_cols: list[str], add_one: bool = True
) -> pd.DataFrame:
    """Log2-transform the given Quantity columns (numeric coercion first)."""
    df_out = df.copy()
    numeric = df_out[qty_cols].apply(pd.to_numeric, errors="coerce")
    df_out[qty_cols] = np.log2(numeric + 1.0) if add_one else np.log2(numeric.replace(0, np.nan))
    return df_out


def make_pairwise_scatter_by_group(
    df_prepared: pd.DataFrame,
    group_to_runs: dict[str, list[str]],
    suffix: str,
    outdir: Path,
    color_map: dict[str, str],
    add_one_for_log2: bool = True,
) -> None:
    """Plot pairwise log2(Quantity) scatter panels per group, one row per
    group, annotated with R^2 and a y = x reference line."""
    df_prepared = drop_duplicate_columns(df_prepared, f"{suffix} (pairwise)")
    available_cols = df_prepared.columns.tolist()

    groups_this_batch = build_group_column_dict(group_to_runs, available_cols)
    log_qly_group_summary(suffix, group_to_runs, groups_this_batch, available_cols)

    if not groups_this_batch:
        log.warning("%s: keine qly-Gruppen für Pairwise Scatter gefunden", suffix)
        return

    qty_cols_all = [c for cols in groups_this_batch.values() for c in cols]
    qty_cols_all = list(dict.fromkeys(qty_cols_all))
    df_log2 = log2_transform_qty(df_prepared, qty_cols_all, add_one=add_one_for_log2)

    n_groups = len(groups_this_batch)
    fig, axes = plt.subplots(n_groups, 3, figsize=(15, 5 * n_groups), squeeze=False)

    for row_idx, (sample, qty_cols) in enumerate(groups_this_batch.items()):
        qty_cols = list(dict.fromkeys(qty_cols))

        log.info("%s | Pairwise | %s | gültige qly-Replikate=%d", suffix, sample, len(qty_cols))

        if len(qty_cols) < 2:
            for ax in axes[row_idx]:
                ax.set_visible(False)
            log.warning("%s | Pairwise | %s: <2 Replikate, Zeile ausgeblendet", suffix, sample)
            continue

        pairs = list(combinations(range(len(qty_cols)), 2))

        for col_idx in range(3):
            ax = axes[row_idx, col_idx]

            if col_idx >= len(pairs):
                ax.set_visible(False)
                continue

            i, j = pairs[col_idx]
            c_i = qty_cols[i]
            c_j = qty_cols[j]

            if c_i not in df_log2.columns or c_j not in df_log2.columns:
                ax.text(0.5, 0.5, "missing column", ha="center", va="center", transform=ax.transAxes)
                ax.set_title(f"{sample}: R{i + 1} vs R{j + 1}", fontsize=9)
                continue

            x = df_log2[c_i]
            y = df_log2[c_j]

            if isinstance(x, pd.DataFrame):
                x = x.iloc[:, 0]
            if isinstance(y, pd.DataFrame):
                y = y.iloc[:, 0]

            mask = x.notna() & y.notna()
            x_vals = x[mask].to_numpy()
            y_vals = y[mask].to_numpy()

            log.info(
                "%s | Pairwise | %s | %s vs %s | gemeinsame Punkte=%d",
                suffix, sample, c_i, c_j, len(x_vals)
            )

            if len(x_vals) < 2:
                ax.text(0.5, 0.5, f"too few points\nn={len(x_vals)}", ha="center", va="center", transform=ax.transAxes)
                ax.set_title(f"{sample}: R{i + 1} vs R{j + 1}", fontsize=9)
                ax.set_xlabel(f"R{i + 1} log₂(Qly)", fontsize=8)
                ax.set_ylabel(f"R{j + 1} log₂(Qly)", fontsize=8)
                continue

            try:
                r, _ = stats.pearsonr(x_vals, y_vals)
                r2 = r ** 2
            except Exception:
                r2 = np.nan

            clr = color_map.get(sample, color_map.get("Unknown", "#333333"))
            ax.scatter(x_vals, y_vals, alpha=0.2, s=5, color=clr)

            lim_min = min(np.nanmin(x_vals), np.nanmin(y_vals))
            lim_max = max(np.nanmax(x_vals), np.nanmax(y_vals))
            ax.plot([lim_min, lim_max], [lim_min, lim_max], "r--", lw=1, label="y=x (ideal)")

            title_color = "black"
            if np.isfinite(r2):
                title_color = "darkgreen" if r2 >= 0.98 else "red"

            title_text = f"{sample}: R{i + 1} vs R{j + 1}"
            title_text += f"\nR² = {r2:.4f}" if np.isfinite(r2) else "\nR² = NA"

            ax.set_xlabel(f"R{i + 1} log₂(Qly)", fontsize=8)
            ax.set_ylabel(f"R{j + 1} log₂(Qly)", fontsize=8)
            ax.set_title(title_text, fontsize=9, color=title_color)
            ax.legend(fontsize=7, loc="best")

    plt.suptitle(f"Pairwise Scatter for replicates [{suffix}]", fontsize=12, fontweight="bold", y=1.0)
    plt.tight_layout(rect=[0, 0, 1, 0.98])

    outdir.mkdir(exist_ok=True, parents=True)
    out_path = outdir / f"pairwise_scatter_{suffix.lstrip('_')}.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    log.info("Pairwise Scatter geschrieben: %s", out_path)


def main() -> None:
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    if not TEMPLATE_PATH.exists():
        log.info("%s nicht gefunden -> erzeuge Template einmalig.", TEMPLATE_PATH)
        df_new = create_random_proteomics_table(
            n_proteins=800, group_to_runs=GROUP_TO_RUNS, missing_rate=0.08, seed=42
        )
        save_template(df_new)

    df = load_template()
    log.info("Template geladen: %s (%d Proteine, %d Spalten)", TEMPLATE_PATH.resolve(), len(df), df.shape[1])

    make_pairwise_scatter_by_group(
        df_prepared=df,
        group_to_runs=GROUP_TO_RUNS,
        suffix="proteomics_template",
        outdir=output_dir,
        color_map=COLOR_MAP,
        add_one_for_log2=True,
    )

    out_path = output_dir / "pairwise_scatter_proteomics_template.png"
    assert out_path.exists() and out_path.stat().st_size > 0, "PNG wurde nicht erzeugt"
    print(f"\nPairwise-Scatter-Plot gespeichert unter:\n{out_path.resolve()}")


if __name__ == "__main__":
    main()