### Overview
- **Purpose**: Visualize set overlaps, intersections, and unique elements across multiple sample cohorts, differentially expressed gene lists, detected proteomics features, or enriched biological pathways.
- **Use Case**: Comparing candidate biomarkers across 2–4 conditions (Venn diagrams) or scalable multi-group intersection discovery across 5–20+ sample cohorts (UpSet plots).
- **Prerequisites**: 
	* Python 3.9+
	* Input formatted as a dictionary of sets: `{"Cohort_A": {"Gene1", "Gene2", ...}, ...}`
- **Data Types**: 
	* Set-like identifiers (Gene symbols, Ensembl IDs, UniProt accessions, Metabolite IDs, Taxa names)
- **Output**:
	* Four-way Venn diagram (`venny4.png`)
	* Scalable multi-sample UpSet plot (`upset_10_random_samples.png`)

#### When to Use

| Situation / Set Count | Recommended Visualization | Python Library | Note |
| :--- | :--- | :--- | :--- |
| **2–4 Cohorts / Sets** | Four-Way Venn Diagram | `venny4py` | Intuitive, classic publication figure for pairwise and multi-way intersections |
| **$\ge 5$ Cohorts / Sets** | UpSet Plot | `upsetplot` | Scalable matrix-layout; overcomes the visual clutter and unreadable intersections of 5+-set Venns |
| **Complex Intersect Querying** | Python `set` operations | `set.intersection()`, `set.union()` | Programmatic extraction of shared core biomarkers |

#### Decision Criteria
- **Use Venn Diagrams when**: Comparing up to 4 biological conditions (e.g., Control, Low Dose, High Dose, Vehicle).
- **Use UpSet Plots when**: Comparing 5 or more experimental groups (e.g., multi-tissue atlas, 10-patient longitudinal profile).
- **Critical rule**: Ensure biological identifiers are harmonized (do not mix UniProt IDs and Gene Symbols in the same set comparison).

### Python Libraries & Methods

| Aspect | `venny4py` | `upsetplot` |
| :--- | :--- | :--- |
| **Main Visualization** | 2–4 Way Venn Diagrams | Scalable Intersection UpSet Barplot |
| **Recommended Group Count** | 2–4 sets | 5–30+ sets |
| **Installation** | `pip install venny4py` | `pip install upsetplot` |
| **Input Format** | `dict[str, set[str]]` | `from_contents(dict[str, set[str]])` |
| **Output Type** | PNG figure files | Matplotlib Figure / Subplot Axes |

### Quick Start Code

```bash
python -m pip install venny4py upsetplot matplotlib pandas
```

```python
import random
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from upsetplot import UpSet, from_contents
from venny4py.venny4py import venny4py


def generate_random_sets_4way(seed: int = 42) -> dict[str, set[str]]:
    """Generates 4 simulated biomarker candidate sets with shared core and specific hits."""
    rng = random.Random(seed)
    universe = [f"Protein_{i+1:03d}" for i in range(200)]
    
    # Shared biological core
    core = set(rng.sample(universe, 25))
    available = list(set(universe) - core)
    
    sets = {}
    for name in ["Control_vs_Mild", "Control_vs_Severe", "Treated_Arm_A", "Treated_Arm_B"]:
        extra = set(rng.sample(available, rng.randint(30, 60)))
        sets[name] = core | extra
    return sets


def generate_random_samples_10way(n_samples: int = 10, seed: int = 42) -> dict[str, set[str]]:
    """Generates 10 simulated patient sample feature sets for UpSet plotting."""
    rng = random.Random(seed)
    universe = [f"Gene_{i+1:04d}" for i in range(1000)]
    shared_core = set(rng.sample(universe, 40))
    available = list(set(universe) - shared_core)
    
    samples = {}
    for i in range(1, n_samples + 1):
        extra = set(rng.sample(available, rng.randint(80, 180)))
        samples[f"Sample_{i:02d}"] = shared_core | extra
    return samples


def main() -> None:
    # 1. Vier-Gruppen Venn-Diagramm (venny4py)
    sets_4way = generate_random_sets_4way()
    out_dir_venn = Path("venn_4_output")
    out_dir_venn.mkdir(exist_ok=True)
    
    venny4py(sets=sets_4way, out=str(out_dir_venn), ext="png", dpi=300)
    plt.close("all")
    
    # Kopiere Ergebnisdatei auf Standardnamen
    gen_file = list(out_dir_venn.glob("*.png"))
    if gen_file:
        venn_dest = Path("venny4.png").resolve()
        import shutil
        shutil.copy2(gen_file[0], venn_dest)
        print(f"✓ 4-Way Venn-Diagramm gespeichert: {venn_dest}")
        
    # 2. Zehn-Proben UpSet Plot (upsetplot)
    samples_10way = generate_random_samples_10way()
    upset_data = from_contents(samples_10way)
    
    upset = UpSet(
        upset_data,
        subset_size="count",
        show_counts=True,
        sort_by="cardinality",
        sort_categories_by="cardinality",
        min_subset_size=5,
        max_subset_rank=20,
        facecolor="#2C7FB8"
    )
    
    fig = plt.figure(figsize=(15, 8))
    upset.plot(fig=fig)
    
    upset_dest = Path("upset_10_random_samples.png").resolve()
    fig.savefig(upset_dest, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"✓ 10-Sample UpSet Plot gespeichert: {upset_dest}")
    
    core_all = set.intersection(*samples_10way.values())
    print(f"\nGemeinsamer Core über alle 10 Samples: {len(core_all)} Gene")


if __name__ == "__main__":
    main()
```

output example:
![[venny4.png]]
![[upset_10_random_samples.png]]
