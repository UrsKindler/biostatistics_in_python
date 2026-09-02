# Further Visualizations (Modular Split Reference)

The visualization functions originally collected in this document have been separated into dedicated, production-grade biostatistics modules with interactive Jupyter Notebooks, standalone scripts, and DIA-NN proteomics examples:

1. **Differential Expression Volcano Plots**:
   👉 [`biostatistics_modules/17_volcano_plots/`](biostatistics_modules/17_volcano_plots/)
   - Volcano plots with $\\log_2 \\text{FC}$ vs $-\\log_{10}(p_{\\text{adj}})$, significance cut-offs, and top biomarker labels.

2. **MA Plots (Ratio vs. Abundance Diagnostics)**:
   👉 [`biostatistics_modules/18_ma_plots/`](biostatistics_modules/18_ma_plots/)
   - Bland-Altman style ratio-intensity plots to diagnose abundance-dependent biases and heteroscedasticity.

3. **Hierarchical Clustered Heatmaps**:
   👉 [`biostatistics_modules/19_clustered_heatmaps/`](biostatistics_modules/19_clustered_heatmaps/)
   - Two-dimensional hierarchical clustering, row Z-score scaling, and multi-group condition annotation bars.

4. **Venn Diagrams & UpSet Overlaps**:
   👉 [`biostatistics_modules/16_venn_diagrams_upset_plots/`](biostatistics_modules/16_venn_diagrams_upset_plots/)
   - 2-to-5 set Venn diagrams, UpSet plots for high-dimensional set overlaps, and region summary tables.

5. **Multivariate Ordination & Scree Plots**:
   👉 [`biostatistics_modules/11_pca_pcoa_nmds_ordination/`](biostatistics_modules/11_pca_pcoa_nmds_ordination/)
   👉 [`biostatistics_modules/05b_nipals_no_imputation_pca/`](biostatistics_modules/05b_nipals_no_imputation_pca/)
