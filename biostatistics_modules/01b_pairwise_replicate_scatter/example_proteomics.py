from __future__ import annotations

import importlib
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

MODULE_DIR = Path(__file__).parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from Random_Proteomics_Dataset_Generator import (
    COLOR_MAP,
    GROUP_TO_RUNS,
    create_random_proteomics_table,
    load_template,
    save_template,
)

mod = importlib.import_module("01b_pairwise_replicate_scatter")
make_pairwise_scatter_by_group = mod.make_pairwise_scatter_by_group


def run_proteomics_example() -> None:
    try:
        df = load_template()
    except Exception:
        df = create_random_proteomics_table(n_proteins=800, group_to_runs=GROUP_TO_RUNS, seed=42)
        save_template(df)

    out_dir = Path(__file__).parent
    out_path = make_pairwise_scatter_by_group(
        df_prepared=df,
        group_to_runs=GROUP_TO_RUNS,
        suffix="proteomics_template",
        outdir=out_dir,
        color_map=COLOR_MAP,
        add_one_for_log2=True,
    )
    print(f"Proteomics Pairwise Scatter successfully generated at: {out_path.resolve()}")


if __name__ == "__main__":
    run_proteomics_example()
