further_voisualisations

def plot_volcano(
    df: pd.DataFrame,
    contrast_name: str,
    outdir: Path,
    padj_thr: float,
    log2fc_thr: float,
    top_n: int,
    gene_map: dict[str, str],
) -> None:
    d = df.copy()

    plt.figure(figsize=(9, 7))

    categories = [("Up", "red"), ("Down", "blue"), ("Not significant", "grey")]
    for label, color in categories:
        mask = d["regulation"] == label
        plt.scatter(
            d.loc[mask, "log2FoldChange"],
            d.loc[mask, "neg_log10_padj"],
            c=color,
            s=20,
            alpha=0.6,
            label=label,
        )

    top_up = d[d["regulation"] == "Up"].sort_values(
        ["padj", "log2FoldChange"], ascending=[True, False]
    ).head(top_n)
    top_down = d[d["regulation"] == "Down"].sort_values(
        ["padj", "log2FoldChange"], ascending=[True, True]
    ).head(top_n)
    top_labels = pd.concat([top_up, top_down])

    mapping = map_gene_labels(top_labels.index.tolist(), gene_map)

    for gene_id, row in top_labels.iterrows():
        display_name = mapping.get(gene_id, gene_id)
        plt.annotate(
            display_name,
            xy=(row["log2FoldChange"], row["neg_log10_padj"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
        )

    plt.axvline(log2fc_thr, color="black", linestyle="--", linewidth=0.8)
    plt.axvline(-log2fc_thr, color="black", linestyle="--", linewidth=0.8)
    plt.axhline(-np.log10(padj_thr), color="black", linestyle="--", linewidth=0.8)

    n_up = int((d["regulation"] == "Up").sum())
    n_down = int((d["regulation"] == "Down").sum())

    plt.text(
        0.02,
        0.98,
        f"Up: {n_up}\nDown: {n_down}",
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
    )

    plt.xlabel("log2 Fold Change")
    plt.ylabel("-log10(padj)")
    plt.title(f"Volcano plot: {contrast_name}")
    plt.legend(frameon=True, fontsize=8)

    ax = plt.gca()
    ax.set_facecolor("white")
    plt.gcf().patch.set_facecolor("white")

    plt.tight_layout()
    plt.savefig(outdir / f"Volcano_{contrast_name}.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_ma(
    df: pd.DataFrame,
    contrast_name: str,
    outdir: Path,
    log2fc_thr: float,
) -> None:
    d = df.copy()

    plt.figure(figsize=(8, 7))
    categories = [("Up", "red"), ("Down", "blue"), ("Not significant", "grey")]
    for label, color in categories:
        mask = d["regulation"] == label
        plt.scatter(
            d.loc[mask, "baseMean"].clip(lower=1e-6),
            d.loc[mask, "log2FoldChange"],
            c=color,
            s=12,
            alpha=0.6,
            label=label,
        )

    plt.xscale("log")
    plt.axhline(0, color="black", linewidth=0.8)
    plt.axhline(log2fc_thr, color="black", linestyle="--", linewidth=0.6)
    plt.axhline(-log2fc_thr, color="black", linestyle="--", linewidth=0.6)

    plt.xlabel("baseMean")
    plt.ylabel("log2 Fold Change")
    plt.title(f"MA plot: {contrast_name}")
    plt.legend(loc="upper right", frameon=True, fontsize=8)

    ax = plt.gca()
    ax.set_facecolor("white")
    plt.gcf().patch.set_facecolor("white")

    plt.tight_layout()
    plt.savefig(outdir / f"MA_{contrast_name}.png", dpi=300, bbox_inches="tight")
    plt.close()


def style_venn(v, colors: list[str]) -> None:
    for patch_id, color in zip(["100", "010", "001"], colors):
        patch = v.get_patch_by_id(patch_id)
        if patch:
            patch.set_facecolor(color)
            patch.set_alpha(0.5)
            patch.set_edgecolor("black")
            patch.set_linewidth(0.8)

    for patch_id in ["110", "101", "011", "111"]:
        patch = v.get_patch_by_id(patch_id)
        if patch:
            patch.set_facecolor("#bbbbbb")
            patch.set_alpha(0.4)
            patch.set_edgecolor("black")
            patch.set_linewidth(0.8)

    for patch_id in ["100", "010", "001", "110", "101", "011", "111"]:
        label = v.get_label_by_id(patch_id)
        if label:
            label.set_fontsize(8)

    if v.set_labels:
        for label in v.set_labels:
            if label:
                label.set_fontsize(8)


def plot_venn_three(
    sets_dict: Dict[str, set],
    title: str,
    outfile: Path,
    colors: list[str],
) -> None:
    keys = list(sets_dict.keys())
    if len(keys) != 3:
        log.warning("Venn benötigt genau 3 Sets, gefunden: %d", len(keys))
        return

    plt.figure(figsize=(6, 6))
    v = venn3(
        [sets_dict[keys[0]], sets_dict[keys[1]], sets_dict[keys[2]]],
        set_labels=tuple(keys),
    )
    style_venn(v, colors=colors)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close()


def summarize_deg_counts(deg_tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for contrast_name, df in deg_tables.items():
        rows.append({
            "contrast": contrast_name,
            "n_tested": df.shape[0],
            "n_significant": int(df["is_significant"].sum()),
            "n_up": int((df["regulation"] == "Up").sum()),
            "n_down": int((df["regulation"] == "Down").sum()),
            "n_not_significant": int((df["regulation"] == "Not significant").sum()),
        })
    return pd.DataFrame(rows)


def summarize_venn_regions(sets_dict: Dict[str, set], set_type: str) -> pd.DataFrame:
    keys = list(sets_dict.keys())
    a, b, c = sets_dict[keys[0]], sets_dict[keys[1]], sets_dict[keys[2]]

    summary = {
        "set_type": set_type,
        f"{keys[0]}_only": len(a - b - c),
        f"{keys[1]}_only": len(b - a - c),
        f"{keys[2]}_only": len(c - a - b),
        f"{keys[0]}_{keys[1]}": len((a & b) - c),
        f"{keys[0]}_{keys[2]}": len((a & c) - b),
        f"{keys[1]}_{keys[2]}": len((b & c) - a),
        "all_three": len(a & b & c),
    }
    return pd.DataFrame([summary])


def add_replicate_column(meta: pd.DataFrame, counts_columns: pd.Index) -> pd.DataFrame:
    meta = meta.copy()
    counts_set = set(pd.Index(counts_columns).astype(str).str.strip())

    if "replicate_name" in meta.columns:
        meta["replicate_name"] = meta["replicate_name"].astype(str).str.strip()
    elif "replicate" in meta.columns:
        meta["replicate_name"] = meta["replicate"].astype(str).str.strip()
    elif {"sample_id", "replicate"}.issubset(meta.columns):
        meta["replicate_name"] = (
            meta["sample_id"].astype(str).str.strip()
            + "_"
            + meta["replicate"].astype(str).str.strip()
        )
    else:
        raise ValueError(
            "Meta braucht 'replicate_name' oder 'replicate' oder die Kombination aus 'sample_id' und 'replicate'."
        )

    meta["replicate_name"] = meta["replicate_name"].astype(str).str.strip()
    meta["replicate_in_counts"] = meta["replicate_name"].isin(counts_set)

    return meta


def build_heatmap_matrix(
    counts: pd.DataFrame,
    meta: pd.DataFrame,
    selected_genes: List[str],
    color_map: dict[str, str],
    allowed_groups: List[str] | None = None,
    use_all_samples: bool = False,
) -> tuple[pd.DataFrame, pd.Series]:
    genes_present = [g for g in selected_genes if g in counts.index]
    if len(genes_present) == 0:
        raise ValueError("Keine ausgewählten Gene in counts gefunden.")

    meta = meta.copy()
    meta["group"] = meta["group"].astype(str).map(sanitize_group_name)
    meta["sample_id"] = meta["sample_id"].astype(str).map(sanitize_group_name)

    counts_cols = counts.columns.astype(str).str.strip()
    meta = meta[meta["sample_id"].isin(counts_cols)].copy()

    if not use_all_samples and allowed_groups is not None:
        allowed_groups = [sanitize_group_name(x) for x in allowed_groups]
        meta = meta[meta["group"].isin(allowed_groups)].copy()

    if meta.empty:
        raise ValueError("Keine gemeinsamen Samples zwischen Meta und Counts gefunden.")

    sample_names = meta["sample_id"].tolist()
    sub_counts = counts.loc[genes_present, sample_names].copy()

    if sub_counts.shape[1] == 0:
        raise ValueError("Nach Filterung keine Count-Spalten für die Heatmap übrig.")

    log_sub = np.log2(sub_counts + 1)

    row_std = log_sub.std(axis=1).replace(0, np.nan)
    z = log_sub.sub(log_sub.mean(axis=1), axis=0).div(row_std, axis=0).fillna(0)

    group_series = meta.set_index("sample_id").loc[sample_names, "group"]

    missing_groups = sorted(set(group_series) - set(color_map.keys()))
    if missing_groups:
        log.warning(
            "Diese groups fehlen im color_map und werden grau dargestellt: %s",
            missing_groups
        )

    col_colors = group_series.map(
        lambda x: color_map.get(x, color_map.get("Unknown", "#aaaaaa"))
    )

    return z, col_colors


def parse_groups_from_contrast(contrast_name: str) -> list[str]:
    parts = contrast_name.split("_vs_")
    if len(parts) != 2:
        raise ValueError(f"Kontrastname nicht im erwarteten Format '*_vs_*': {contrast_name}")
    return [parts[0].strip(), parts[1].strip()]


def plot_heatmap(
    counts: pd.DataFrame,
    meta: pd.DataFrame,
    df_deg: pd.DataFrame,
    contrast_name: str,
    outdir: Path,
    top_n: int,
    color_map: dict[str, str],
    gene_map: dict[str, str],
) -> None:
    up = df_deg[df_deg["regulation"] == "Up"].sort_values(
        ["padj", "log2FoldChange"], ascending=[True, False]
    ).head(top_n)
    down = df_deg[df_deg["regulation"] == "Down"].sort_values(
        ["padj", "log2FoldChange"], ascending=[True, True]
    ).head(top_n)

    selected_genes = up.index.tolist() + down.index.tolist()
    if len(selected_genes) == 0:
        log.warning("Keine signifikanten Gene für Heatmap in %s", contrast_name)
        return

    mapping = map_gene_labels(selected_genes, gene_map)
    contrast_groups = parse_groups_from_contrast(contrast_name)

    try:
        z, col_colors = build_heatmap_matrix(
            counts=counts,
            meta=meta,
            selected_genes=selected_genes,
            color_map=color_map,
            allowed_groups=contrast_groups,
            use_all_samples=False,
        )
    except ValueError as e:
        log.warning("Heatmap für %s übersprungen: %s", contrast_name, e)
        return

    z = z.loc[[g for g in selected_genes if g in z.index]]
    z.index = [mapping.get(idx, idx) for idx in z.index]

    sns.set(context="talk", style="white")
    g = sns.clustermap(
        z,
        cmap="RdBu_r",
        center=0,
        xticklabels=True,
        yticklabels=True,
        col_colors=col_colors.values,
        cbar_kws={"label": "Z-score"},
        figsize=(14, 12),
        dendrogram_ratio=(0.1, 0.1),
        cbar_pos=(0.88, 0.82, 0.03, 0.14),
    )

    g.ax_heatmap.tick_params(axis="y", labelsize=8)
    g.ax_heatmap.tick_params(axis="x", labelsize=8)
    plt.setp(g.ax_heatmap.get_xticklabels(), rotation=45, ha="right")

    g.fig.patch.set_facecolor("white")
    g.ax_heatmap.set_facecolor("white")

    g.fig.suptitle(
        f"Top {top_n} Up + Top {top_n} Down: {contrast_name}",
        y=0.98,
        fontsize=16,
    )
    g.fig.subplots_adjust(left=0.22, right=0.88, top=0.93, bottom=0.05)

    g.ax_cbar.set_position([0.90, 0.82, 0.02, 0.14])
    g.ax_cbar.set_title("Z-score", fontsize=9, pad=6)
    g.ax_cbar.tick_params(labelsize=8)

    g.savefig(
        outdir / f"Heatmap_top{top_n}up_top{top_n}down_{contrast_name}.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()


def plot_consensus_heatmap(
    counts: pd.DataFrame,
    meta: pd.DataFrame,
    joint_df: pd.DataFrame | None,
    outdir: Path,
    top_n: int,
    color_map: dict[str, str],
    gene_map: dict[str, str],
) -> None:
    if joint_df is None:
        return

    flag_keep = joint_df["flag_category"].isin([1, 2])

    up = joint_df[flag_keep & (joint_df["integrated_log2FC"] > 0)].sort_values(
        by="integrated_log2FC", ascending=False
    ).head(top_n)

    down = joint_df[flag_keep & (joint_df["integrated_log2FC"] < 0)].sort_values(
        by="integrated_log2FC", ascending=True
    ).head(top_n)

    selected_genes = up.index.tolist() + down.index.tolist()
    if len(selected_genes) == 0:
        log.warning("Keine konsistenten Gene für Consensus-Heatmap gefunden.")
        return

    mapping = map_gene_labels(selected_genes, gene_map)

    try:
        z, col_colors = build_heatmap_matrix(
            counts=counts,
            meta=meta,
            selected_genes=selected_genes,
            color_map=color_map,
            allowed_groups=None,
            use_all_samples=True,
        )
    except ValueError as e:
        log.warning("Consensus-Heatmap übersprungen: %s", e)
        return

    z = z.loc[[g for g in selected_genes if g in z.index]]
    z.index = [mapping.get(idx, idx) for idx in z.index]

    sns.set(context="talk", style="white")
    g = sns.clustermap(
        z,
        cmap="RdBu_r",
        center=0,
        xticklabels=True,
        yticklabels=True,
        col_colors=col_colors.values,
        cbar_kws={"label": "Z-score"},
        figsize=(14, 12),
        dendrogram_ratio=(0.1, 0.1),
        cbar_pos=(0.88, 0.82, 0.03, 0.14),
    )

    g.ax_heatmap.tick_params(axis="y", labelsize=8)
    g.ax_heatmap.tick_params(axis="x", labelsize=8)
    plt.setp(g.ax_heatmap.get_xticklabels(), rotation=45, ha="right")

    g.fig.patch.set_facecolor("white")
    g.ax_heatmap.set_facecolor("white")

    g.fig.suptitle(
        f"Consensus genes (Flag1/2), Top {top_n} Up + Top {top_n} Down",
        y=0.98,
        fontsize=16,
    )
    g.fig.subplots_adjust(left=0.22, right=0.88, top=0.93, bottom=0.05)

    g.ax_cbar.set_position([0.90, 0.82, 0.02, 0.14])
    g.ax_cbar.set_title("Z-score", fontsize=9, pad=6)
    g.ax_cbar.tick_params(labelsize=8)

    g.savefig(
        outdir / f"Heatmap_consensus_top{top_n}up_top{top_n}down.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()


    def run_pca(
    groups_subset: list[str],
    groups_qly: dict[str, list[str]],
    filtered_df: pd.DataFrame,
    color_map: dict[str, str],
    title: str,
    filename: str,
    output_dir: Path,
    add_one_for_log2: bool = True,
) -> PCA | None:
    selected_cols: list[str] = []
    sample_to_group: dict[str, str] = {}

    groups_subset = list(dict.fromkeys(groups_subset))

    for group_name in groups_subset:
        cols = groups_qly.get(group_name, [])
        valid_cols = [c for c in cols if c in filtered_df.columns]
        missing_cols = [c for c in cols if c not in filtered_df.columns]

        log.info(
            "PCA | %s | Gruppe %s | qly-Spalten in groups_qly=%d | gültig im DataFrame=%d | fehlend=%d",
            title, group_name, len(cols), len(valid_cols), len(missing_cols)
        )
        if missing_cols:
            log.warning(
                "PCA | %s | Gruppe %s | fehlende Spalten-Beispiele: %s%s",
                title,
                group_name,
                missing_cols[:5],
                " ..." if len(missing_cols) > 5 else ""
            )

        for c in valid_cols:
            selected_cols.append(c)
            sample_to_group[c] = group_name

    selected_cols = list(dict.fromkeys(selected_cols))

    if len(selected_cols) < 3:
        log.warning("PCA übersprungen (%s): zu wenige gültige qly-Spalten", title)
        return None

    X_df = filtered_df.loc[:, selected_cols].copy()
    X_df = X_df.apply(pd.to_numeric, errors="coerce")

    if add_one_for_log2:
        X_df = np.log2(X_df + 1.0)
    else:
        X_df = np.log2(X_df.replace(0, np.nan))

    all_samples_before = X_df.columns.tolist()

    X = X_df.T
    X = X.dropna(axis=0, how="all")
    samples_after_dropna_rows = X.index.tolist()

    dropped_samples = [s for s in all_samples_before if s not in samples_after_dropna_rows]
    if dropped_samples:
        log.warning(
            "PCA | %s | Samples entfernt, weil komplett NA nach Log-Transform: %s%s",
            title,
            dropped_samples[:10],
            " ..." if len(dropped_samples) > 10 else ""
        )

    X = X.dropna(axis=1, how="all")

    if X.shape[0] < 3:
        log.warning("PCA übersprungen (%s): zu wenige Samples nach Filter", title)
        return None

    X = X.apply(lambda col: col.fillna(col.median()), axis=0)
    X = X.fillna(0.0)

    scaler = StandardScaler(with_mean=True, with_std=True)
    X_scaled = scaler.fit_transform(X.values)

    n_comp = min(X_scaled.shape[0], X_scaled.shape[1])
    if n_comp < 2:
        log.warning("PCA übersprungen (%s): weniger als 2 Komponenten möglich", title)
        return None

    pca_full = PCA(n_components=n_comp)
    pcs_full = pca_full.fit_transform(X_scaled)

    pca_df = pd.DataFrame(
        pcs_full[:, :2],
        index=X.index,
        columns=["PC1", "PC2"]
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    plotted = set()

    for sample in pca_df.index:
        row = pca_df.loc[sample]
        group_name = sample_to_group.get(sample, "Unknown")
        clr = color_map.get(group_name, color_map.get("Unknown", "#333333"))
        label = group_name if group_name not in plotted else None

        ax.scatter(
            row["PC1"], row["PC2"],
            s=80, color=clr, edgecolor="black", alpha=0.85, label=label
        )
        plotted.add(group_name)

    ax.axhline(0, color="#cccccc", linewidth=0.7)
    ax.axvline(0, color="#cccccc", linewidth=0.7)
    ax.set_xlabel(f"PC1 ({pca_full.explained_variance_ratio_[0] * 100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({pca_full.explained_variance_ratio_[1] * 100:.1f}% var)")
    ax.set_title(title)

    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=True,
        borderaxespad=0.0,
        fontsize=8
    )

    plt.tight_layout(rect=[0, 0, 0.82, 1])

    out_path = output_dir / filename
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    log.info("PCA geschrieben: %s", out_path)
    return pca_full


def run_screeplot(
    pca_obj: PCA | None,
    title: str,
    filename: str,
    output_dir: Path,
    max_pcs: int = 12,
) -> None:
    if pca_obj is None:
        return

    explained = pca_obj.explained_variance_ratio_
    n_show = min(len(explained), max_pcs)

    explained_show = explained[:n_show]
    pcs = np.arange(1, n_show + 1)

    plt.figure(figsize=(8, 5))
    plt.bar(
        pcs,
        explained_show * 100,
        color="#4C72B0",
        alpha=0.8,
        label="Explained variance (%)"
    )
    plt.plot(
        pcs,
        np.cumsum(explained_show * 100),
        marker="o",
        color="#C44E52",
        label="Cumulative (%)"
    )
    plt.xticks(pcs)
    plt.xlabel("Principal Component")
    plt.ylabel("Explained Variance (%)")
    plt.title(f"{title} (first {n_show} PCs)")
    plt.legend(loc="best", frameon=True)
    plt.tight_layout()

    out_path = output_dir / filename
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    log.info("Scree Plot geschrieben: %s", out_path)