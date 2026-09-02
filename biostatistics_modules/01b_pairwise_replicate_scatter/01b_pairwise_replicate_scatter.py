from __future__ import annotations

import logging
import sys
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from Random_Proteomics_Dataset_Generator import (
    COLOR_MAP,
    GROUP_TO_RUNS,
    TEMPLATE_PATH,
    create_random_proteomics_table,
    load_template,
    save_template,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("pairwise_scatter")


def drop_duplicate_columns(df: pd.DataFrame, context: str) -> pd.DataFrame:
    dup_mask = df.columns.duplicated()
    n_dupes = int(dup_mask.sum())
    if n_dupes:
        log.warning("%s: %d duplicate columns removed", context, n_dupes)
    return df.loc[:, ~dup_mask]


def build_group_column_dict(
    group_to_runs: dict[str, list[str]], available_cols: list[str]
) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for group_name, runs in group_to_runs.items():
        qty_cols = [f"{run}.raw.PG.Quantity" for run in runs]
        valid_cols = [c for c in qty_cols if c in available_cols]
        if valid_cols:
            groups[group_name] = valid_cols
    return groups


def log2_transform_qty(
    df: pd.DataFrame, qty_cols: list[str], add_one: bool = True
) -> pd.DataFrame:
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
) -> Path:
    df_prepared = drop_duplicate_columns(df_prepared, f"{suffix} (pairwise)")
    available_cols = df_prepared.columns.tolist()

    groups_this_batch = build_group_column_dict(group_to_runs, available_cols)
    if not groups_this_batch:
        raise ValueError("No matching quantity columns found for pairwise scatter.")

    qty_cols_all = [c for cols in groups_this_batch.values() for c in cols]
    qty_cols_all = list(dict.fromkeys(qty_cols_all))
    df_log2 = log2_transform_qty(df_prepared, qty_cols_all, add_one=add_one_for_log2)

    n_groups = len(groups_this_batch)
    fig, axes = plt.subplots(n_groups, 3, figsize=(15, 4.5 * n_groups), squeeze=False)

    for row_idx, (sample, qty_cols) in enumerate(groups_this_batch.items()):
        qty_cols = list(dict.fromkeys(qty_cols))
        pairs = list(combinations(range(len(qty_cols)), 2))

        for col_idx in range(3):
            ax = axes[row_idx, col_idx]
            if col_idx >= len(pairs):
                ax.set_visible(False)
                continue

            i, j = pairs[col_idx]
            c_i = qty_cols[i]
            c_j = qty_cols[j]

            x = df_log2[c_i].dropna()
            y = df_log2[c_j].dropna()
            common = df_log2[[c_i, c_j]].dropna()

            if len(common) < 2:
                ax.text(0.5, 0.5, "Too few shared points", ha="center", va="center")
                continue

            x_vals = common[c_i].to_numpy()
            y_vals = common[c_j].to_numpy()

            try:
                r, _ = stats.pearsonr(x_vals, y_vals)
                r2 = r ** 2
            except Exception:
                r2 = np.nan

            clr = color_map.get(sample, "#333333")
            ax.scatter(x_vals, y_vals, alpha=0.25, s=6, color=clr)

            lim_min = min(np.nanmin(x_vals), np.nanmin(y_vals))
            lim_max = max(np.nanmax(x_vals), np.nanmax(y_vals))
            ax.plot([lim_min, lim_max], [lim_min, lim_max], "r--", lw=1.2, label="y = x")

            title_color = "darkgreen" if (np.isfinite(r2) and r2 >= 0.95) else ("darkorange" if (np.isfinite(r2) and r2 >= 0.85) else "red")
            r2_str = f"R² = {r2:.4f}" if np.isfinite(r2) else "R² = NA"

            ax.set_xlabel(f"R{i+1} log₂(Intensity)", fontsize=9)
            ax.set_ylabel(f"R{j+1} log₂(Intensity)", fontsize=9)
            ax.set_title(f"{sample}: R{i+1} vs R{j+1} ({r2_str})", fontsize=10, color=title_color, fontweight="bold")
            ax.legend(fontsize=8, loc="upper left")
            ax.grid(alpha=0.3)

    plt.suptitle(f"Pairwise Replicate Reproducibility [{suffix}]", fontsize=14, fontweight="bold", y=1.0)
    plt.tight_layout(rect=[0, 0, 1, 0.98])

    outdir.mkdir(exist_ok=True, parents=True)
    out_path = outdir / f"pairwise_scatter_{suffix.lstrip('_')}.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    return out_path


def main() -> None:
    output_dir = Path(__file__).parent
    try:
        df = load_template()
    except Exception:
        df = create_random_proteomics_table(n_proteins=800, group_to_runs=GROUP_TO_RUNS, seed=42)
        save_template(df)

    out_path = make_pairwise_scatter_by_group(
        df_prepared=df,
        group_to_runs=GROUP_TO_RUNS,
        suffix="proteomics_template",
        outdir=output_dir,
        color_map=COLOR_MAP,
        add_one_for_log2=True,
    )
    print(f"Pairwise scatter saved to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
