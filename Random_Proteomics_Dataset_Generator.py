from __future__ import annotations

import logging
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("proteomics_generator")

ROOT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT_DIR / "output"
TEMPLATE_PATH = OUTPUT_DIR / "proteomics_template.csv"

GROUP_TO_RUNS: dict[str, list[str]] = {
    "Control": ["Control_R1", "Control_R2", "Control_R3"],
    "Treatment_A": ["TreatmentA_R1", "TreatmentA_R2", "TreatmentA_R3"],
    "Treatment_B": ["TreatmentB_R1", "TreatmentB_R2", "TreatmentB_R3"],
}

COLOR_MAP: dict[str, str] = {
    "Control": "#2C7FB8",
    "Treatment_A": "#D9534F",
    "Treatment_B": "#5CB85C",
    "Unknown": "#333333",
}


def _random_gene_names(rng: np.random.Generator, n: int) -> list[str]:
    return [f"GENE{idx:04d}" for idx in range(1, n + 1)]


def create_random_proteomics_table(
    n_proteins: int = 800,
    group_to_runs: dict[str, list[str]] | None = None,
    missing_rate: float = 0.08,
    seed: int = 42,
) -> pd.DataFrame:
    if group_to_runs is None:
        group_to_runs = GROUP_TO_RUNS

    rng = np.random.default_rng(seed)

    protein_ids = [f"P{idx:05d}" for idx in range(1, n_proteins + 1)]
    genes = _random_gene_names(rng, n_proteins)
    descriptions = [f"Protein description {idx}" for idx in range(1, n_proteins + 1)]
    protein_names = [f"{g}_HUMAN" for g in genes]

    base_log2_abundance = rng.normal(loc=15, scale=2.5, size=n_proteins)

    # 80 distinct differentially abundant proteins (40 Up, 40 Down in Treatment_A vs Control)
    n_diff_ta = 80
    diff_idx_ta = rng.choice(n_proteins, size=n_diff_ta, replace=False)
    effect_ta = np.zeros(n_proteins)
    effect_ta[diff_idx_ta[:40]] = rng.uniform(2.2, 4.2, 40)   # Up
    effect_ta[diff_idx_ta[40:]] = rng.uniform(-4.2, -2.2, 40) # Down

    # Distinct diff proteins for Treatment_B
    diff_idx_tb = rng.choice(n_proteins, size=n_diff_ta, replace=False)
    effect_tb = np.zeros(n_proteins)
    effect_tb[diff_idx_tb[:40]] = rng.uniform(2.0, 3.8, 40)
    effect_tb[diff_idx_tb[40:]] = rng.uniform(-3.8, -2.0, 40)

    df = pd.DataFrame(
        {
            "PG.ProteinAccessions": protein_ids,
            "PG.Genes": genes,
            "PG.ProteinDescriptions": descriptions,
            "PG.ProteinNames": protein_names,
        }
    )

    all_runs = [run for runs in group_to_runs.values() for run in runs]
    quantity_matrix = pd.DataFrame(index=df.index)

    for group_name, runs in group_to_runs.items():
        if group_name == "Treatment_A":
            group_effect = effect_ta
            group_shift = 0.2
        elif group_name == "Treatment_B":
            group_effect = effect_tb
            group_shift = -0.2
        else:
            group_effect = np.zeros(n_proteins)
            group_shift = 0.0

        for run in runs:
            run_noise = rng.normal(loc=0, scale=0.15, size=n_proteins)
            replicate_noise = rng.normal(loc=0, scale=0.10, size=n_proteins)

            log2_abundance = base_log2_abundance + group_shift + group_effect + run_noise + replicate_noise
            quantity = np.power(2, log2_abundance)
            quantity_matrix[run] = quantity

    for run in all_runs:
        missing_prob = np.clip(
            (18 - base_log2_abundance) / 20 * missing_rate * 2.5, 0.0, 0.30
        )
        missing_mask = rng.random(n_proteins) < missing_prob
        quantity_matrix.loc[missing_mask, run] = np.nan

    n_precursors_base = rng.integers(2, 15, size=n_proteins)
    n_peptides_base = rng.integers(1, 8, size=n_proteins)

    for run in all_runs:
        df[f"{run}.raw.PG.Quantity"] = quantity_matrix[run]
        df[f"{run}.raw.PG.NrOfPrecursorsMeasured"] = np.clip(
            n_precursors_base + rng.integers(-1, 2, size=n_proteins), 0, 30
        )
        df[f"{run}.raw.PG.NrOfStrippedSequencesIdentified"] = np.clip(
            n_peptides_base + rng.integers(0, 2, size=n_proteins), 0, 15
        )

    # Compute p-values for reference
    ctrl_cols = [f"{r}.raw.PG.Quantity" for r in group_to_runs["Control"]]
    ta_cols = [f"{r}.raw.PG.Quantity" for r in group_to_runs["Treatment_A"]]

    c_mat = np.log2(df[ctrl_cols].replace(0, np.nan))
    t_mat = np.log2(df[ta_cols].replace(0, np.nan))
    log2fc = t_mat.mean(axis=1) - c_mat.mean(axis=1)

    from scipy import stats
    p_vals = []
    for idx in range(n_proteins):
        c = c_mat.iloc[idx].dropna()
        t = t_mat.iloc[idx].dropna()
        if len(c) >= 2 and len(t) >= 2:
            _, p = stats.ttest_ind(t, c, equal_var=True)
            p_vals.append(p)
        else:
            p_vals.append(np.nan)

    df["PG.Log2FC"] = log2fc
    df["PG.Pvalue"] = p_vals

    return df


def save_template(df: pd.DataFrame, path: Path | None = None) -> Path:
    target = path or TEMPLATE_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(target, index=False)
    log.info("Saved proteomics template with shape %s to %s", df.shape, target)
    return target


def load_template(path: Path | None = None) -> pd.DataFrame:
    source = path or TEMPLATE_PATH
    if not source.exists():
        log.warning("Template not found at %s. Creating new default table.", source)
        df = create_random_proteomics_table()
        save_template(df, source)
        return df
    log.info("Loaded template from %s", source)
    return pd.read_csv(source)


def main() -> None:
    df = create_random_proteomics_table(n_proteins=800, seed=42)
    out_path = save_template(df)
    print(f"Template successfully generated at: {out_path.resolve()}")


if __name__ == "__main__":
    main()
