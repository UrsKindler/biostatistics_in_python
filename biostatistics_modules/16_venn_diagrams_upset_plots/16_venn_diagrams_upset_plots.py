from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib_venn import venn3
from upsetplot import UpSet, from_indicators


def plot_venn_and_upset(sets_dict: dict[str, set], outpath: Path) -> None:
    keys = list(sets_dict.keys())

    fig, ax = plt.subplots(figsize=(8, 8))
    v = venn3([sets_dict[keys[0]], sets_dict[keys[1]], sets_dict[keys[2]]], set_labels=tuple(keys), ax=ax)

    # Style 3-way Venn
    colors = ["#2C7FB8", "#D9534F", "#5CB85C"]
    for patch_id, color in zip(["100", "010", "001"], colors):
        p = v.get_patch_by_id(patch_id)
        if p:
            p.set_facecolor(color)
            p.set_alpha(0.6)

    ax.set_title("3-Way Set Intersection (Venn Diagram)", fontweight="bold", fontsize=14)
    plt.tight_layout()
    outpath.parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


def summarize_venn_regions(sets_dict: dict[str, set], set_type: str = "Proteins") -> pd.DataFrame:
    keys = list(sets_dict.keys())
    a, b, c = sets_dict[keys[0]], sets_dict[keys[1]], sets_dict[keys[2]]
    return pd.DataFrame([{
        "Set Type": set_type,
        f"{keys[0]} Only": len(a - b - c),
        f"{keys[1]} Only": len(b - a - c),
        f"{keys[2]} Only": len(c - a - b),
        f"{keys[0]} & {keys[1]}": len((a & b) - c),
        f"{keys[0]} & {keys[2]}": len((a & c) - b),
        f"{keys[1]} & {keys[2]}": len((b & c) - a),
        "All Three": len(a & b & c),
    }])


def main() -> None:
    set_a = {f"Gene_{i}" for i in range(1, 100)}
    set_b = {f"Gene_{i}" for i in range(40, 140)}
    set_c = {f"Gene_{i}" for i in range(70, 180)}

    sets_dict = {"Control": set_a, "Treatment_A": set_b, "Treatment_B": set_c}
    summary_df = summarize_venn_regions(sets_dict)
    print("Venn Overlap Summary:")
    print(summary_df)

    out_dir = Path(__file__).parent
    outpath = out_dir / "16_venn_upset_plots.png"
    plot_venn_and_upset(sets_dict, outpath)
    print(f"Venn diagram saved to: {outpath.resolve()}")


if __name__ == "__main__":
    main()
