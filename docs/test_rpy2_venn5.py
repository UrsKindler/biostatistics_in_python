#!/usr/bin/env python3

import os
import random
from pathlib import Path


# ---------------------------------------------------------------------
# R-Konfiguration für Windows:
# Nur erforderlich, falls R_HOME nicht bereits im VS-Code-Terminal
# gesetzt wurde. Passe den Pfad an, falls sich deine R-Version ändert.
# ---------------------------------------------------------------------

R_HOME = r"C:\Program Files\R\R-4.6.1"

os.environ.setdefault("R_HOME", R_HOME)
os.environ["PATH"] = rf"{R_HOME}\bin\x64;{os.environ['PATH']}"


from rpy2 import robjects as ro
from rpy2.robjects.vectors import StrVector


def create_random_sets(
    universe_size: int = 1000,
    shared_core_size: int = 40,
    seed: int = 42,
) -> dict[str, set[str]]:
    """
    Erzeugt fünf simulierte Proteinmengen.

    Alle Gruppen enthalten einen gemeinsamen Kern von Proteinen.
    Zusätzlich besitzt jede Gruppe zufällig gezogene Proteine aus
    demselben Hintergrund-Universum.
    """
    rng = random.Random(seed)

    universe = [f"Protein_{i:04d}" for i in range(1, universe_size + 1)]

    shared_core = set(rng.sample(universe, shared_core_size))
    remaining_proteins = list(set(universe) - shared_core)

    target_sizes = {
        "WC": 83,
        "Glucose": 294,
        "HDL_C": 104,
        "TG": 111,
        "BP": 53,
    }

    protein_sets: dict[str, set[str]] = {}

    for group_name, total_size in target_sizes.items():
        n_extra = total_size - shared_core_size

        if n_extra > len(remaining_proteins):
            raise ValueError(
                f"Nicht genügend Proteine für {group_name}: "
                f"{n_extra} zusätzliche Proteine benötigt."
            )

        extra_proteins = set(rng.sample(remaining_proteins, n_extra))
        protein_sets[group_name] = shared_core | extra_proteins

    return protein_sets


def main() -> None:
    protein_sets = create_random_sets()

    print("Protein-Anzahl pro Gruppe:")
    for group_name, proteins in protein_sets.items():
        print(f"{group_name}: {len(proteins)} Proteine")

    common_to_all = set.intersection(*protein_sets.values())
    print(
        f"\nGemeinsam in allen fünf Gruppen: "
        f"{len(common_to_all)} Proteine"
    )

    output_file = Path("venn_5_random_groups.png").resolve()

    r_protein_sets = ro.ListVector(
        {
            group_name: StrVector(sorted(proteins))
            for group_name, proteins in protein_sets.items()
        }
    )

    ro.globalenv["protein_sets"] = r_protein_sets
    ro.globalenv["output_file"] = str(output_file)

    ro.r(
        r"""
        suppressPackageStartupMessages(library(VennDiagram))

        venn.diagram(
          x = protein_sets,

          filename = output_file,

          imagetype = "png",
          height = 2800,
          width = 3200,
          resolution = 300,

          category.names = c(
            "WC",
            "Glucose",
            "HDL-C",
            "TG",
            "BP"
          ),

          fill = c(
            "#4C9F70",
            "#8FC3E8",
            "#F29CA3",
            "#C7D957",
            "#FDB863"
          ),

          alpha = rep(0.55, 5),
          col = rep("black", 5),
          lwd = rep(1.2, 5),

          cat.col = c(
            "#2B6A48",
            "#2879A7",
            "#A83A45",
            "#758715",
            "#B96418"
          ),

          cat.cex = rep(1.5, 5),
          cex = rep(0.8, 31),

          margin = 0.10,

          disable.logging = TRUE
        )
        """
    )

    if not output_file.exists() or output_file.stat().st_size == 0:
        raise RuntimeError(
            "Das Venn-Diagramm wurde nicht erzeugt. "
            "Prüfe R_HOME sowie die Installation von VennDiagram."
        )

    print(f"\nVenn-Diagramm erfolgreich erzeugt:\n{output_file}")


if __name__ == "__main__":
    main()