#!/usr/bin/env python3

import random
from pathlib import Path

import matplotlib

# Nicht-interaktives Backend:
# erzeugt eine PNG-Datei ohne problematisches Tkinter-Fenster
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from upsetplot import UpSet, from_contents


def create_random_samples(
    n_samples: int = 10,
    universe_size: int = 1000,
    min_features: int = 100,
    max_features: int = 300,
    shared_core_size: int = 40,
    seed: int = 42,
) -> dict[str, set[str]]:
    """
    Erstellt zufällige Feature-Mengen für n_samples.

    Jedes Sample enthält:
    - einen global geteilten Kern gemeinsamer Features,
    - zusätzliche zufällige Features aus einem gemeinsamen Universum.
    """
    if shared_core_size > min_features:
        raise ValueError(
            "shared_core_size darf nicht größer als min_features sein."
        )

    if max_features > universe_size:
        raise ValueError(
            "max_features darf nicht größer als universe_size sein."
        )

    rng = random.Random(seed)

    universe = [f"Gene_{i:04d}" for i in range(1, universe_size + 1)]
    shared_core = set(rng.sample(universe, shared_core_size))
    available_features = list(set(universe) - shared_core)

    sample_sets: dict[str, set[str]] = {}

    for i in range(1, n_samples + 1):
        target_size = rng.randint(min_features, max_features)
        n_extra = target_size - shared_core_size

        if n_extra > len(available_features):
            raise ValueError(
                f"Nicht genügend Features für Sample_{i:02d}: "
                f"{n_extra} zusätzliche Features angefordert, "
                f"aber nur {len(available_features)} verfügbar."
            )

        extra_features = set(rng.sample(available_features, n_extra))
        sample_sets[f"Sample_{i:02d}"] = shared_core | extra_features

    return sample_sets


def main() -> None:
    sample_sets = create_random_samples(
        n_samples=10,
        universe_size=1000,
        min_features=100,
        max_features=300,
        shared_core_size=40,
        seed=42,
    )

    print("Feature-Anzahl pro Sample:")
    for sample_name, features in sample_sets.items():
        print(f"{sample_name}: {len(features)} Features")

    common_to_all = set.intersection(*sample_sets.values())
    print(f"\nGemeinsam in allen 10 Samples: {len(common_to_all)} Features")

    upset_data = from_contents(sample_sets)

    upset = UpSet(
        upset_data,
        subset_size="count",
        show_counts=False,
        sort_by="cardinality",
        sort_categories_by="cardinality",
        min_subset_size=3,
        max_subset_rank=30,
        facecolor="#2C7FB8",
    )

    fig = plt.figure(figsize=(16, 9))
    upset.plot(fig=fig)

    output_file = Path("upset_10_random_samples.png").resolve()

    fig.savefig(
        output_file,
        dpi=300,
        facecolor="white",
    )

    plt.close(fig)

    print(f"\nUpSet-Plot gespeichert unter:\n{output_file}")


if __name__ == "__main__":
    main()