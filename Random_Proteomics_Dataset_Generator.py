"""
00. Random Proteomics Dataset Generator (shared template)
============================================================
Purpose: Generate one reproducible, DIA-NN/Spectronaut-style proteomics
report table that serves as the common input template for all downstream
scripts (01a Data Quality, 01b Pairwise Scatter, 02 PCA, ...).

Use Case: Provide a realistic wide-format protein quantity table with
per-run metadata columns, so every visualization/statistics script can be
tested against the exact same structure your real DIA-NN/Spectronaut
export would have.

Prerequisites: pandas, numpy
Data Types: Protein-level identifiers (accessions, gene symbols) plus
per-sample quantitative columns (NrOfStrippedSequencesIdentified,
NrOfPrecursorsIdentified, Quantity) for n runs, replicated across groups.

Output: CSV file (proteomics_template.csv) written once and re-used
(read back, never regenerated inline) by every downstream script.

Column schema (matches the real export naming convention):
    PG.ProteinAccessions, PG.Genes, PG.ProteinDescriptions, PG.ProteinNames,
    PG.Pvalue, PG.Qvalue,
    <run>.raw.PG.NrOfStrippedSequencesIdentified,
    <run>.raw.PG.NrOfPrecursorsIdentified,
    <run>.raw.PG.Quantity
    for every run name in RUN_NAMES.

IMPORTANT: If you edit GROUP_TO_RUNS, n_proteins, missing_rate, or seed
for a follow-up analysis, re-run main() and commit the resulting CSV --
every downstream script (01a, 01b, 02, ...) reads this persisted file
instead of generating its own random data, so all figures stay
comparable and reproducible across scripts and re-runs.
"""

from pathlib import Path

import numpy as np
import pandas as pd

TEMPLATE_PATH = Path("output") / "proteomics_template.csv"

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
    """
    Build a synthetic wide-format proteomics table with realistic
    structure: shared protein identity columns plus per-run quantitative
    columns, log-normal intensities, batch-correlated replicates within a
    group, injected missingness (MAR-like: low-abundance proteins missing
    more often) and a handful of differentially abundant proteins between
    groups.
    """
    if group_to_runs is None:
        group_to_runs = GROUP_TO_RUNS

    rng = np.random.default_rng(seed)

    protein_ids = [f"P{idx:05d}" for idx in range(1, n_proteins + 1)]
    genes = _random_gene_names(rng, n_proteins)
    descriptions = [f"Protein description {idx}" for idx in range(1, n_proteins + 1)]
    protein_names = [f"{g}_HUMAN" for g in genes]

    base_log2_abundance = rng.normal(loc=15, scale=2.5, size=n_proteins)

    n_diff = max(1, int(0.05 * n_proteins))
    diff_idx = rng.choice(n_proteins, size=n_diff, replace=False)
    diff_effect = rng.choice([-1, 1], size=n_diff) * rng.uniform(1.5, 3.0, size=n_diff)

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
        group_shift = {"Control": 0.0, "Treatment_A": 0.3, "Treatment_B": -0.2}.get(group_name, 0.0)
        for run in runs:
            run_noise = rng.normal(loc=0, scale=0.4, size=n_proteins)
            replicate_noise = rng.normal(loc=0, scale=0.25, size=n_proteins)

            log2_abundance = base_log2_abundance + group_shift + run_noise + replicate_noise

            if group_name != "Control":
                log2_abundance[diff_idx] = log2_abundance[diff_idx] + diff_effect

            quantity = np.power(2, log2_abundance)
            quantity_matrix[run] = quantity

    for run in all_runs:
        missing_prob = np.clip(
            (18 - base_log2_abundance) / 20 * missing_rate * 3, 0.0, 0.35
        )
        missing_mask = rng.random(n_proteins) < missing_prob
        quantity_matrix.loc[missing_mask, run] = np.nan

    n_precursors_base = rng.integers(2, 15, size=n_proteins)
    n_peptides_base = rng.integers(1, 8, size=n_proteins)

    for run in all_runs:
        is_missing = quantity_matrix[run].isna()

        n_precursors = n_precursors_base + rng.integers(-1, 2, size=n_proteins)
        n_precursors = np.clip(n_precursors, 0, None)
        n_precursors[is_missing.to_numpy()] = 0

        n_peptides = n_peptides_base + rng.integers(-1, 2, size=n_proteins)
        n_peptides = np.clip(n_peptides, 0, None)
        n_peptides[is_missing.to_numpy()] = 0

        df[f"{run}.raw.PG.NrOfStrippedSequencesIdentified"] = n_peptides
        df[f"{run}.raw.PG.NrOfPrecursorsIdentified"] = n_precursors
        df[f"{run}.raw.PG.Quantity"] = quantity_matrix[run].round(2)

    p_values = rng.uniform(0, 1, n_proteins)
    p_values[diff_idx] = rng.uniform(0, 0.01, n_diff)
    df["PG.Pvalue"] = p_values.round(6)

    sorted_p = np.sort(p_values)
    ranks = np.searchsorted(sorted_p, p_values) + 1
    q_values = np.clip(p_values * n_proteins / ranks, 0, 1)
    df["PG.Qvalue"] = q_values.round(6)

    ordered_cols = [
        "PG.ProteinAccessions", "PG.Genes", "PG.ProteinDescriptions",
        "PG.ProteinNames", "PG.Pvalue", "PG.Qvalue",
    ]
    for run in all_runs:
        ordered_cols += [
            f"{run}.raw.PG.NrOfStrippedSequencesIdentified",
            f"{run}.raw.PG.NrOfPrecursorsIdentified",
            f"{run}.raw.PG.Quantity",
        ]

    return df[ordered_cols]


def save_template(df: pd.DataFrame, path: Path = TEMPLATE_PATH) -> Path:
    """Persist the random dataset once so every downstream script reads
    the identical CSV instead of regenerating random data independently."""
    path.parent.mkdir(exist_ok=True, parents=True)
    df.to_csv(path, index=False)
    return path


def load_template(path: Path = TEMPLATE_PATH) -> pd.DataFrame:
    """Load the shared template CSV. Raises if it has not been generated
    yet via main()."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} nicht gefunden. Bitte zuerst dieses Script ausfuehren, "
            "um die Template-CSV zu erzeugen."
        )
    return pd.read_csv(path)


def main() -> None:
    df = create_random_proteomics_table(
        n_proteins=800,
        group_to_runs=GROUP_TO_RUNS,
        missing_rate=0.08,
        seed=42,
    )

    out_path = save_template(df)

    print(f"Proteine: {len(df)}")
    print(f"Spalten total: {df.shape[1]}")
    print(f"Gruppen -> Runs: {GROUP_TO_RUNS}")
    print(f"\nTemplate-CSV gespeichert unter:\n{out_path.resolve()}")
    print("\n=== df.head() ===")
    print(df.head())


if __name__ == "__main__":
    main()
