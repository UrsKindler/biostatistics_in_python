from venny4py.venny4py import venny4py
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib


sample_sets = {
    "Sample_01": {"Protein_A", "Protein_B", "Protein_C", "Protein_D"},
    "Sample_02": {"Protein_B", "Protein_C", "Protein_E"},
    "Sample_03": {"Protein_A", "Protein_C", "Protein_F"},
    "Sample_04": {"Protein_B", "Protein_D", "Protein_G"},
}

venny4py(
    sets=sample_sets,
    out="venn_4_samples",
    ext="png",
    dpi=300,
)

output_file = Path("venny4.png").resolve()

plt.savefig(
    output_file,
    dpi=300,
    facecolor="white",
)