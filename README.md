# Biostatistics in Python: End-to-End Workflow & Reference Guide

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Latest-orange.svg)](https://scikit-learn.org/)
[![SciPy](https://img.shields.io/badge/SciPy-Statistics-blue.svg)](https://scipy.org/)
[![Statsmodels](https://img.shields.io/badge/Statsmodels-Inference-yellow.svg)](https://www.statsmodels.org/)

A rigorous, modular framework for modern **biostatistics, high-throughput omics diagnostics, multivariate ordination, and machine learning in Python**. Features direct **R-to-Python translations**, statistical decision trees, Postdoc-level theoretical foundations, interactive Jupyter notebooks, and a shared synthetic mass-spectrometry proteomics pipeline (DIA-NN / Spectronaut compatible).

---

## Table of Contents

- [Workflow Pipeline & Architecture](#-workflow-pipeline--architecture)
- [Statistical Decision Matrix (Cheat Sheet)](#-statistical-decision-matrix-cheat-sheet)
- [Phase 1: Data Quality & Preprocessing](#-phase-1-data-quality--preprocessing)
  - [01. Data Quality Assessment](#01-data-quality-assessment)
  - [01b. Pairwise Replicate Reproducibility Scatter](#01b-pairwise-replicate-reproducibility-scatter)
  - [02. Missing Data Handling & Imputation](#02-missing-data-handling--imputation)
  - [03. Abundance-Threshold & Variance Filtering](#03-abundance-threshold--variance-filtering)
  - [04. Normalization & Feature Scaling](#04-normalization--feature-scaling)
- [Phase 2: Statistical Assumption Testing](#-phase-2-statistical-assumption-testing)
  - [05. Shapiro-Wilk & Normality Diagnostics](#05-shapiro-wilk--normality-diagnostics)
  - [05b. NIPALS No-Imputation PCA for Incomplete Matrices](#05b-nipals-no-imputation-pca-for-incomplete-matrices)
  - [02b. Variance Homogeneity & Homoscedasticity](#02b-variance-homogeneity--homoscedasticity)
- [Phase 3: Hypothesis Testing & Association](#-phase-3-hypothesis-testing--association)
  - [06. Pearson & Spearman Correlation](#06-pearson--spearman-correlation)
  - [07. Two-Group Comparisons: Welch / Student t-Test & Mann-Whitney U-Test](#07-two-group-comparisons-welch--student-t-test--mann-whitney-u-test)
  - [08. Multi-Group Comparisons: One-Way ANOVA & Kruskal-Wallis](#08-multi-group-comparisons-one-way-anova--kruskal-wallis)
- [Phase 4: Post-Hoc Tests, Effect Sizes & Multiple Testing](#-phase-4-post-hoc-tests-effect-sizes--multiple-testing)
  - [09_1. Pairwise Post-Hoc Tests (Tukey HSD, Games-Howell, Dunn)](#09_1-pairwise-post-hoc-tests-tukey-hsd-games-howell-dunn)
  - [09_2. Effect Size Quantification (Cohen's d, Hedges' g, Eta-Squared)](#09_2-effect-size-quantification-cohens-d-hedges-g-eta-squared)
  - [09_3. Multiple Testing Correction & False Discovery Rate (Postdoc Level)](#09_3-multiple-testing-correction--false-discovery-rate-postdoc-level)
- [Phase 5: Multivariate Statistics & Ordination](#-phase-5-multivariate-statistics--ordination)
  - [09. ANOSIM & PERMANOVA Distance Permutations](#09-anosim--permanova-distance-permutations)
  - [11. Multivariate Ordination: PCA, PCoA & NMDS](#11-multivariate-ordination-pca-pcoa--nmds)
- [Phase 6: Unsupervised Learning & Clustering](#-phase-6-unsupervised-learning--clustering)
  - [10. K-Means, Hierarchical Clustering (Ward) & DBSCAN](#10-k-means-hierarchical-clustering-ward--dbscan)
  - [19. Hierarchical Clustered Heatmaps & Biomarker Modules](#19-hierarchical-clustered-heatmaps--biomarker-modules)
- [Phase 7: Differential Expression & Diagnostic Omics Visualizations](#-phase-7-differential-expression--diagnostic-omics-visualizations)
  - [17. Differential Expression Volcano Plots](#17-differential-expression-volcano-plots)
  - [18. MA Plots (Abundance vs. Ratio Bias Diagnostics)](#18-ma-plots-abundance-vs-ratio-bias-diagnostics)
- [Phase 8: Supervised Machine Learning & Validation](#-phase-8-supervised-machine-learning--validation)
  - [12. Decision Trees & Multi-Layer Perceptrons (MLP)](#12-decision-trees--multi-layer-perceptrons-mlp)
  - [15. Random Forests & Biomarker Importance](#15-random-forests--biomarker-importance)
  - [13. Cross-Validation Strategies (Stratified K-Fold, LOOCV)](#13-cross-validation-strategies-stratified-k-fold-loocv)
  - [14. Model Evaluation & Performance Metrics (ROC-AUC, PR-AUC)](#14-model-evaluation--performance-metrics-roc-auc-pr-auc)
- [Phase 9: Set Overlaps & High-Dimensional Enrichment](#-phase-9-set-overlaps--high-dimensional-enrichment)
  - [16. Venn Diagrams (2–5 Sets) & Scalable UpSet Plots](#16-venn-diagrams-25-sets--scalable-upset-plots)
- [Synthetic Proteomics Pipeline](#-synthetic-proteomics-pipeline)
- [R vs. Python Rosetta Stone Table](#-r-vs-python-rosetta-stone-table)
- [Installation & Environment Setup](#-installation--environment-setup)
- [Repository Structure](#-repository-structure)
- [License](#-license)

---

## Workflow Pipeline & Architecture

![Biostatistics Workflow Pipeline](statistical%20scripts.png)

The complete end-to-end biostatistics and machine learning execution workflow:

```mermaid
flowchart TD
    A["Raw High-Throughput Omics Data
(Proteomics DIA-NN/Spectronaut, RNA-seq, Metabolomics)"] --> B["Phase 1: Preprocessing & Quality Control
- Data Quality Assessment & Missing Patterns
- Pairwise Replicate Reproducibility (R²)
- Missing Data Imputation (k-NN, Median, MissForest)
- Abundance & Prevalence Threshold Filtering
- Normalization (Median, Quantile, Z-Score)"]
    
    B --> C{"Phase 2: Assumption Testing
- Normality (Shapiro-Wilk, Q-Q Plots)
- Homogeneity of Variance (Levene, Bartlett)
- Native Missingness PCA (NIPALS)"}
    
    C -->|"Parametric Assumptions Satisfied"| D["Parametric Comparative Pipeline
- Pearson Correlation (r)
- Two-Sample Student / Welch t-Test
- One-Way ANOVA (F-Test)
- Tukey HSD / Games-Howell Post-Hoc"]
    
    C -->|"Non-Parametric / Distribution Free"| E["Non-Parametric Comparative Pipeline
- Spearman Rank Correlation (ρ)
- Mann-Whitney U-Test / Wilcoxon
- Kruskal-Wallis H-Test
- Dunn Post-Hoc with FDR Adjustment"]
    
    D --> F["Phase 4: Multiplicity & Effect Size Control
- Effect Sizes (Cohen's d, Hedges' g, Eta²)
- Multiple Testing Correction (FWER: Holm, Hochberg)
- False Discovery Rate (BH, BY, Storey q-value, locFDR)"]
    E --> F
    
    B --> G["Phase 5: Multivariate Ordination
- Distance Metrics (Bray-Curtis, Euclidean, Jaccard)
- Permutational Testing (PERMANOVA, ANOSIM)
- Ordination (Linear PCA, Metric PCoA, Non-Metric NMDS)"]
    
    B --> H["Phase 6: Cluster Discovery & Heatmaps
- Unsupervised Clustering (K-Means, Ward, DBSCAN)
- Silhouette Validation & Scree Analysis
- Hierarchical Clustered Heatmaps (Z-Score Standardized)"]
    
    F --> J["Phase 7: Differential Expression Diagnostics
- Volcano Plots (log2FC vs -log10 padj)
- MA Diagnostic Plots (baseMean vs log2FC)"]
    
    B --> K["Phase 8: Supervised Machine Learning
- Decision Trees & Multi-Layer Perceptrons
- Random Forests (Gini & Permutation Importance)
- Stratified K-Fold & Repeated CV
- Model Evaluation (ROC-AUC, PR-AUC, Confusion Matrix)"]
    
    F --> L["Phase 9: Set Intersection Analytics
- 2- to 5-Way Venn Diagrams
- High-Dimensional UpSet Plots & Disjoint Region Counts"]
```

---

## Statistical Decision Matrix (Cheat Sheet)

| Research Objective / Data Topology | Distributional Assumption | Variance Homogeneity | Recommended Method | Python Implementation |
| :--- | :--- | :--- | :--- | :--- |
| **Compare 2 Independent Groups** | Gaussian ($p > 0.05$) | Homoscedastic ($\sigma_1^2 = \sigma_2^2$) | **Student's Two-Sample t-Test** | `scipy.stats.ttest_ind(equal_var=True)` |
| **Compare 2 Independent Groups** | Gaussian ($p > 0.05$) | Heteroscedastic ($\sigma_1^2 \neq \sigma_2^2$) | **Welch's t-Test** | `scipy.stats.ttest_ind(equal_var=False)` |
| **Compare 2 Independent Groups** | Non-Gaussian / Skewed | Any | **Mann-Whitney U-Test** | `scipy.stats.mannwhitneyu()` |
| **Compare 2 Paired Samples** | Gaussian differences | Not applicable | **Paired t-Test** | `scipy.stats.ttest_rel()` |
| **Compare 2 Paired Samples** | Non-Gaussian differences | Not applicable | **Wilcoxon Signed-Rank Test** | `scipy.stats.wilcoxon()` |
| **Compare $\ge 3$ Independent Groups** | Gaussian ($p > 0.05$) | Homoscedastic ($p > 0.05$) | **One-Way ANOVA** | `scipy.stats.f_oneway()` |
| **Compare $\ge 3$ Independent Groups** | Non-Gaussian / Ordinal | Any | **Kruskal-Wallis H-Test** | `scipy.stats.kruskal()` |
| **Pairwise Post-Hoc Comparison** | Gaussian (Post-ANOVA) | Equal Variances | **Tukey's HSD** | `statsmodels.stats.multicomp.pairwise_tukeyhsd()` |
| **Pairwise Post-Hoc Comparison** | Gaussian (Post-ANOVA) | Unequal Variances | **Games-Howell Test** | `scikit_posthocs.posthoc_gameshowell()` |
| **Pairwise Post-Hoc Comparison** | Non-Gaussian (Post-Kruskal) | Any | **Dunn's Test with FDR** | `scikit_posthocs.posthoc_dunn(p_adjust="fdr_bh")` |
| **Linear Association (2 Continuous)** | Bivariate Normal | Linear Relationship | **Pearson Correlation ($r$)** | `scipy.stats.pearsonr()` |
| **Monotonic Association (2 Continuous/Ordinal)** | Non-Gaussian / Monotone | Monotonic Relationship | **Spearman Rank Correlation ($\rho$)** | `scipy.stats.spearmanr()` |
| **Multivariate Group Separation** | Non-parametric | Distance Matrix | **PERMANOVA (Adonis)** | `skbio.stats.distance.permanova()` |
| **Multivariate Rank Separation** | Non-parametric | Distance Matrix | **ANOSIM** | `skbio.stats.distance.anosim()` |
| **Dimensionality Reduction (Complete Data)** | Continuous | Standardized | **PCA (SVD)** | `sklearn.decomposition.PCA()` |
| **Dimensionality Reduction (Missing Data)** | Incomplete matrix (NaNs) | Standardized | **NIPALS PCA (No-Imputation)** | `05b_nipals_pca.perform_nipals_pca()` |
| **Non-Euclidean Ordination** | Arbitrary Distance | Metric | **PCoA (Metric MDS)** | `sklearn.manifold.MDS(metric=True)` |
| **Non-Linear Rank Ordination** | Distance Rank Order | Non-metric | **NMDS** | `sklearn.manifold.MDS(metric=False)` |
| **Multiple Testing Correction (FWER)** | Simultaneous tests | Arbitrary / Positive Dep. | **Holm-Bonferroni** | `multipletests(method="holm")` |
| **Multiple Testing Correction (FDR)** | Omics Screening ($m \sim 10^4$) | PRDS Dependence | **Benjamini-Hochberg (BH)** | `multipletests(method="fdr_bh")` |
| **Positive FDR / q-value Estimation** | High Signal Density ($m_1 \gg 0$) | Uniform Null Tail | **Storey's $q$-value ($\hat{\pi}_0$ Spline)** | `09_3_fdr.calculate_storey_qvalues()` |
| **Set Overlaps ($2 \le k \le 4$ Groups)** | Categorical feature sets | Finite Sets | **Venn Diagram** | `matplotlib_venn.venn3()` / `venny4py` |
| **Set Overlaps ($k \ge 5$ Groups)** | High-dimensional sets | Arbitrary Sets | **UpSet Plot** | `upsetplot.UpSet()` |

---

## Phase 1: Data Quality & Preprocessing

### 01. Data Quality Assessment
- **Objective**: Systematic diagnostic profiling of missingness mechanisms (MCAR, MAR, MNAR), measurement noise, skewness, and extreme multivariate outliers prior to downstream inferential statistics.
- **Decision Criteria**:
  - Outlier detection via the Interquartile Range ($IQR = Q_3 - Q_1$): datapoints outside $[Q_1 - 1.5 \cdot IQR, Q_3 + 1.5 \cdot IQR]$ flagged.
  - Missingness topology visualization using missing-value heatmaps and nullity dendrograms.
- **Module**: [`biostatistics_modules/01_data_quality_assessment/`](biostatistics_modules/01_data_quality_assessment/)
- **Visual Output**: `01_data_quality_assessment.png`

```python
import missingno as msno
import pandas as pd

df = pd.read_csv("data.csv", index_col=0)
msno.matrix(df)

Q1, Q3 = df.quantile(0.25), df.quantile(0.75)
IQR = Q3 - Q1
outliers = ((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).sum()
```

---

### 01b. Pairwise Replicate Reproducibility Scatter
- **Objective**: Assess technical and biological reproducibility across experimental replicates within conditions via multi-panel scatter grids, identity lines ($y = x$), and coefficients of determination ($R^2$).
- **Decision Criteria**:
  - $R^2 \ge 0.95$: Excellent technical reproducibility (DIA-NN / Spectronaut MS quality standard).
  - $R^2 < 0.85$: Flagged for potential injection failure, column clogging, or batch effect.
- **Module**: [`biostatistics_modules/01b_pairwise_replicate_scatter/`](biostatistics_modules/01b_pairwise_replicate_scatter/)
- **Visual Output**: `pairwise_scatter_proteomics_template.png`

```python
from scipy import stats

def compute_replicate_r2(s1, s2):
    common = pd.concat([s1, s2], axis=1).dropna()
    r, _ = stats.pearsonr(common.iloc[:, 0], common.iloc[:, 1])
    return r ** 2
```

---

### 02. Missing Data Handling & Imputation
- **Objective**: Principled imputation of missing values aligned with the underlying missingness mechanism.
- **Imputation Strategy**:
  - **Complete Case Analysis (Listwise deletion)**: Permissible only under MCAR with $< 5\%$ loss.
  - **Median / Mode Imputation**: Fast baseline for MAR with low missingness ($< 10\%$).
  - **k-Nearest Neighbors (k-NN Imputer)**: Preserves feature-feature covariance structures ($k=5$).
  - **MissForest / Iterative Imputer**: Multivariate chained equations (MICE) for non-linear relationships.
- **Module**: [`biostatistics_modules/02_missing_data_handling/`](biostatistics_modules/02_missing_data_handling/)
- **Visual Output**: `02_missing_data_handling.png`

```python
from sklearn.impute import KNNImputer

imputer = KNNImputer(n_neighbors=5, weights="distance")
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns, index=df.index)
```

---

### 03. Abundance-Threshold & Variance Filtering
- **Objective**: Mitigate technical noise and alleviate the severe multiple testing burden by filtering out low-abundance, low-prevalence, and near-zero variance features.
- **Criteria**:
  - **Prevalence Filter**: Feature present in $\ge X\%$ (e.g., 50%) of replicates in at least one biological group.
  - **Variance Threshold**: Remove non-informative features with $\mathrm{Var}(X) \le \epsilon$.
- **Module**: [`biostatistics_modules/03_abundance_threshold_filtering/`](biostatistics_modules/03_abundance_threshold_filtering/)
- **Visual Output**: `03_abundance_filtering.png`

---

### 04. Normalization & Feature Scaling
- **Objective**: Eliminate systematic loading differences, total ion chromatogram (TIC) variations, and scale disparities.
- **Techniques**:
  - **Median Centering / Total Sum Normalization**: Corrects for global sample loading differences.
  - **Quantile Normalization**: Forces identical empirical distributions across all runs.
  - **Z-Score Standardization**: Centers mean to $\mu = 0$ and scales variance to $\sigma^2 = 1$.
  - **$\log_2(x + 1)$ Transformation**: Stabilizes variance across the dynamic range.
- **Module**: [`biostatistics_modules/04_normalization_zscore_scaling/`](biostatistics_modules/04_normalization_zscore_scaling/)
- **Visual Output**: `04_normalization_and_scaling.png`

---

## Phase 2: Statistical Assumption Testing

### 05. Shapiro-Wilk & Normality Diagnostics
- **Objective**: Rigorous empirical verification of Gaussian distribution assumptions to select between parametric and non-parametric statistical tests.
- **Diagnostics**:
  - $N < 50$: **Shapiro-Wilk Test** ($W$-statistic, optimal statistical power).
  - $N \ge 50$: **D'Agostino-Pearson Omnibus $K^2$** and **Anderson-Darling Test**.
  - Visual validation via Normal Q-Q probability plots.
- **Module**: [`biostatistics_modules/05_shapiro_wilk_normality_test/`](biostatistics_modules/05_shapiro_wilk_normality_test/)
- **Visual Output**: `05_shapiro_wilk_normality.png`

```python
from scipy import stats

stat, p_val = stats.shapiro(data.dropna())
is_normal = p_val > 0.05
```

---

### 05b. NIPALS No-Imputation PCA for Incomplete Matrices
- **Objective**: Execute Principal Component Analysis (PCA) directly on matrices containing missing values (NaNs) **without prior imputation**.
- **Mathematical Mechanism**:
  The **NIPALS** (Nonlinear Iterative Partial Least Squares / Wold 1966) algorithm estimates score vectors $t_k$ and loading vectors $p_k$ iteratively using only observed values:
  $$p_k = rac{\sum_{i \in \mathrm{obs}} x_{ij} t_{i,k}}{\sum_{i \in \mathrm{obs}} t_{i,k}^2}, \quad t_{i,k} = rac{\sum_{j \in \mathrm{obs}} x_{ij} p_{j,k}}{\sum_{j \in \mathrm{obs}} p_{j,k}^2}$$
  Followed by deflation $X_{k+1} = X_k - t_k p_k^T$.
- **Module**: [`biostatistics_modules/05b_nipals_no_imputation_pca/`](biostatistics_modules/05b_nipals_no_imputation_pca/)
- **Visual Output**: `05b_nipals_no_imputation_pca.png`

---

### 02b. Variance Homogeneity & Homoscedasticity
- **Objective**: Test equality of variances across groups ($\sigma_1^2 = \sigma_2^2 = \dots = \sigma_k^2$).
- **Test Selection**:
  - **Levene's Test (Median / Brown-Forsythe)**: Robust against non-normality.
  - **Bartlett's Test**: High power under strict normality.
  - **Fligner-Killeen Test**: Non-parametric rank-based test for heavily skewed data.
- **Module**: [`biostatistics_modules/02_variance_homogeneity_tests/`](biostatistics_modules/02_variance_homogeneity_tests/)
- **Visual Output**: `02_variance_homogeneity_tests.png`

---

## Phase 3: Hypothesis Testing & Association

### 06. Pearson & Spearman Correlation
- **Objective**: Quantify linear ($r$) or monotonic ($ho$) associations between pairs of continuous/ordinal biological variables.
- **Formulas**:
  - **Pearson $r$**: $r = rac{\sum (x_i - ar{x})(y_i - ar{y})}{\sqrt{\sum (x_i - ar{x})^2 \sum (y_i - ar{y})^2}}$
  - **Spearman $ho$**: Pearson $r$ applied to ranked variables $\mathrm{rank}(X), \mathrm{rank}(Y)$.
- **Module**: [`biostatistics_modules/06_pearson_spearman_correlation/`](biostatistics_modules/06_pearson_spearman_correlation/)
- **Visual Output**: `06_correlation_analysis.png`

---

### 07. Two-Group Comparisons: Welch / Student t-Test & Mann-Whitney U-Test
- **Objective**: Evaluate location parameter differences between two biological conditions.
- **Workflow**:
  - Normal & Equal Variances $\implies$ **Student's Two-Sample t-Test**.
  - Normal & Unequal Variances $\implies$ **Welch's t-Test** (Satterthwaite degrees of freedom).
  - Non-Normal / Skewed $\implies$ **Mann-Whitney U-Test** (Rank-Sum Test).
- **Module**: [`biostatistics_modules/07_ttest_mann_whitney_utest/`](biostatistics_modules/07_ttest_mann_whitney_utest/)
- **Visual Output**: `07_two_group_comparisons.png`

---

### 08. Multi-Group Comparisons: One-Way ANOVA & Kruskal-Wallis
- **Objective**: Test for omnibus differences across $\ge 3$ experimental cohorts while controlling nominal Type I error rates.
- **Methods**:
  - **One-Way ANOVA ($F$-Test)**: $F = rac{\mathrm{MS}_{\mathrm{between}}}{\mathrm{MS}_{\mathrm{within}}}$.
  - **Kruskal-Wallis $H$-Test**: Non-parametric evaluation of mean ranks across $k$ groups.
- **Module**: [`biostatistics_modules/08_anova_kruskal_wallis_test/`](biostatistics_modules/08_anova_kruskal_wallis_test/)
- **Visual Output**: `08_anova_kruskal_wallis.png`

---

## Phase 4: Post-Hoc Tests, Effect Sizes & Multiple Testing

### 09_1. Pairwise Post-Hoc Tests (Tukey HSD, Games-Howell, Dunn)
- **Objective**: Pinpoint exactly which group pairs drive statistical significance following a significant omnibus test.
- **Selection**:
  - Post-ANOVA (Equal Variances) $\implies$ **Tukey's Honestly Significant Difference (HSD)** (Studentized range $q$).
  - Post-ANOVA (Unequal Variances) $\implies$ **Games-Howell Test**.
  - Post-Kruskal-Wallis $\implies$ **Dunn's Test** with Benjamini-Hochberg FDR adjustments.
- **Module**: [`biostatistics_modules/09_1_posthoc_tests/`](biostatistics_modules/09_1_posthoc_tests/)
- **Visual Output**: `09_1_posthoc_tests.png`

---

### 09_2. Effect Size Quantification (Cohen's d, Hedges' g, Eta-Squared)
- **Objective**: Quantify biological and clinical magnitude of change independent of sample size $N$.
- **Key Metrics**:
  - **Cohen's $d$**: $d = rac{ar{x}_1 - ar{x}_2}{s_{\mathrm{pooled}}}$ (Thresholds: $0.2 = 	ext{Small}, 0.5 = 	ext{Medium}, 0.8 = 	ext{Large}$).
  - **Hedges' $g$**: Small-sample bias corrected estimator $g pprox d \cdot \left(1 - rac{3}{4(n_1 + n_2) - 9}ight)$.
  - **Eta-Squared ($\eta^2$) & Partial $\eta^2$**: Proportion of variance accounted for by treatment in ANOVA.
- **Module**: [`biostatistics_modules/09_2_effect_size/`](biostatistics_modules/09_2_effect_size/)
- **Visual Output**: `09_2_effect_size_analysis.png`

---

### 09_3. Multiple Testing Correction & False Discovery Rate (Postdoc Level)
- **Objective**: Control statistical error rates during simultaneous testing of $m \sim 10^3 - 10^6$ omics hypotheses.
- **Theoretical Foundations**:
  1. **Family-Wise Error Rate (FWER)**:
     - **Bonferroni (1936)**: $	ilde{p}_i = \min(1, m \cdot p_i)$ (Union bound, ultra-conservative).
     - **Holm-Bonferroni (1979)**: Step-down $	ilde{p}_{(i)} = \min(1, \max_{j \le i} (m - j + 1) p_{(j)})$ (Uniformly more powerful).
     - **Hochberg (1988) / Hommel (1988)**: Step-up procedure under Simes inequality.
     - **Westfall-Young (1993)**: Permutation-based step-down ($	ext{max}T / 	ext{min}P$).
  2. **False Discovery Rate (FDR)**:
     - **Benjamini-Hochberg (BH 1995)**: Step-up $k = \max \{ i : p_{(i)} \le rac{i}{m} lpha \}$. Controls $	ext{FDR} \le rac{m_0}{m}lpha \le lpha$ under PRDS.
     - **Benjamini-Yekutieli (BY 2001)**: Controls FDR under arbitrary dependence using harmonic sum $c(m) = \sum_{i=1}^m rac{1}{i}$.
  3. **Storey's Positive FDR & $q$-value (Storey 2002, 2003)**:
     - Estimates true null proportion $\hat{\pi}_0(\lambda) = rac{\#\{p_i > \lambda\}}{m(1 - \lambda)}$ via natural cubic splines.
     - Calculates $q(p_{(i)}) = \min_{j \ge i} \left( rac{\hat{\pi}_0 \cdot m \cdot p_{(j)}}{j} ight)$, improving statistical power by factor $rac{1}{\hat{\pi}_0}$.
  4. **Local FDR ($	ext{locFDR}$, Efron 2004)**:
     - Posterior probability of being a false discovery given test statistic $z$: $	ext{locFDR}(z) = rac{\pi_0 f_0(z)}{f(z)}$.
  5. **Covariate-Modulated FDR**: Independent Hypothesis Weighting (IHW, Ignatiadis et al. 2016).
- **Module**: [`biostatistics_modules/09_3_false_discovery_rate/`](biostatistics_modules/09_3_false_discovery_rate/)
- **Visual Output**: `09_3_false_discovery_rate.png` (4-panel diagnostic benchmark)

---

## Phase 5: Multivariate Statistics & Ordination

### 09. ANOSIM & PERMANOVA Distance Permutations
- **Objective**: Non-parametric hypothesis testing of whole-community or multivariate omics profiles between experimental conditions.
- **Methods**:
  - **PERMANOVA (Permutational Multivariate ANOVA)**: Partitions dissimilarity matrices (Bray-Curtis, Euclidean) and calculates pseudo-$F$ statistics across 999+ permutations.
  - **ANOSIM (Analysis of Similarities)**: Compares ranked distances between vs. within groups ($R \in [-1, 1]$; $R > 0.75 \implies$ strong group separation).
- **Module**: [`biostatistics_modules/09_anosim_permanova/`](biostatistics_modules/09_anosim_permanova/)
- **Visual Output**: `09_multivariate_anosim_permanova.png`

---

### 11. Multivariate Ordination: PCA, PCoA & NMDS
- **Objective**: Linear and non-linear dimensionality reduction and unsupervised 2D/3D ordination.
- **Algorithms**:
  - **PCA**: SVD-based linear orthogonal projection maximizing explained variance.
  - **PCoA (Metric MDS)**: Classical scaling preserving metric distances from non-Euclidean dissimilarity matrices.
  - **NMDS**: Iterative non-metric rank preservation (Kruskal stress $< 0.1 \implies$ excellent ordination).
- **Module**: [`biostatistics_modules/11_pca_pcoa_nmds_ordination/`](biostatistics_modules/11_pca_pcoa_nmds_ordination/)
- **Visual Output**: `11_multivariate_ordination.png`

---

## Phase 6: Unsupervised Learning & Clustering

### 10. K-Means, Hierarchical Clustering (Ward) & DBSCAN
- **Objective**: Unsupervised discovery of patient subgroups, biomarker co-expression modules, and disease endotypes.
- **Algorithms**:
  - **K-Means**: Centroid-based partitioning optimized via Silhouette Score and Elbow analysis.
  - **Agglomerative Hierarchical (Ward's Minimum Variance)**: Builds hierarchical tree structures minimizing within-cluster sum of squares.
  - **DBSCAN**: Density-based spatial clustering capable of finding arbitrary non-convex clusters and isolating technical outliers.
- **Module**: [`biostatistics_modules/10_kmeans_hierarchical_dbscan_clustering/`](biostatistics_modules/10_kmeans_hierarchical_dbscan_clustering/)
- **Visual Output**: `10_unsupervised_clustering.png`

---

### 19. Hierarchical Clustered Heatmaps & Biomarker Modules
- **Objective**: Multi-condition quantitative visualization of expression profiles with simultaneous two-dimensional clustering (row biomarker clustering + column sample dendrograms) and categorical metadata annotations.
- **Module**: [`biostatistics_modules/19_clustered_heatmaps/`](biostatistics_modules/19_clustered_heatmaps/)
- **Visual Output**: `19_clustered_heatmaps.png`

---

## Phase 7: Differential Expression & Diagnostic Omics Visualizations

### 17. Differential Expression Volcano Plots
- **Objective**: Bivariate visualization combining biological magnitude ($\log_2 	ext{Fold Change}$) on the x-axis and statistical significance ($-\log_{10} p_{\mathrm{adj}}$) on the y-axis to isolate bona fide biomarker candidates.
- **Module**: [`biostatistics_modules/17_volcano_plots/`](biostatistics_modules/17_volcano_plots/)
- **Visual Output**: `17_volcano_plots.png`

---

### 18. MA Plots (Abundance vs. Ratio Bias Diagnostics)
- **Objective**: Bland-Altman ratio-intensity plots ($M = \log_2(R/G)$ vs $A = rac{1}{2}(\log_2 R + \log_2 G)$) to diagnose abundance-dependent non-linear biases and variance heteroscedasticity across the dynamic range.
- **Module**: [`biostatistics_modules/18_ma_plots/`](biostatistics_modules/18_ma_plots/)
- **Visual Output**: `18_ma_plots.png`

---

## Phase 8: Supervised Machine Learning & Validation

### 12. Decision Trees & Multi-Layer Perceptrons (MLP)
- **Objective**: Supervised disease state classification and interpretable rule extraction.
- **Models**:
  - **Decision Trees (`DecisionTreeClassifier`)**: White-box non-linear decision splits.
  - **Multi-Layer Perceptron (`MLPClassifier`)**: Neural feed-forward architecture with backpropagation and ReLU activation.
- **Module**: [`biostatistics_modules/12_decision_trees_neural_networks/`](biostatistics_modules/12_decision_trees_neural_networks/)
- **Visual Output**: `12_decision_trees_and_neural_nets.png`

---

### 15. Random Forests & Biomarker Importance
- **Objective**: High-performance bagging ensemble reducing tree variance and extracting robust feature importances (Mean Decrease Impurity / Gini).
- **Module**: [`biostatistics_modules/15_random_forests/`](biostatistics_modules/15_random_forests/)
- **Visual Output**: `15_random_forests.png`

---

### 13. Cross-Validation Strategies (Stratified K-Fold, LOOCV)
- **Objective**: Prevent data leakage, optimistic performance bias, and overfitting in high-dimensional biomarker selection.
- **Schemes**:
  - **Stratified K-Fold**: Preserves exact class prevalence across all validation folds.
  - **Repeated K-Fold**: Averages variance across multiple random split seeds.
  - **Leave-One-Out (LOOCV)**: Deterministic validation for small clinical pilot cohorts ($N < 30$).
- **Module**: [`biostatistics_modules/13_cross_validation/`](biostatistics_modules/13_cross_validation/)
- **Visual Output**: `13_cross_validation.png`

---

### 14. Model Evaluation & Performance Metrics (ROC-AUC, PR-AUC)
- **Objective**: Holistic evaluation of binary and multi-class diagnostic classifiers.
- **Metrics**:
  - **ROC-AUC**: Area under the Receiver Operating Characteristic curve.
  - **Precision-Recall (PR-AUC)**: Informative evaluation under extreme class imbalance.
  - **Confusion Matrix, Sensitivity (Recall), Specificity, and F1-Score**.
- **Module**: [`biostatistics_modules/14_model_evaluation_metrics/`](biostatistics_modules/14_model_evaluation_metrics/)
- **Visual Output**: `14_model_evaluation_metrics.png`

---

## Phase 9: Set Overlaps & High-Dimensional Enrichment

### 16. Venn Diagrams (2–5 Sets) & Scalable UpSet Plots
- **Objective**: Quantify shared biomarker intersections, unique identifiers, and set overlaps across experimental conditions.
- **Selection**:
  - $2 \le k \le 4$ Sets: Classic Venn diagrams via `matplotlib_venn` / `venny4py`.
  - $k \ge 5$ Sets: Matrix-layout UpSet plots via `upsetplot` to eliminate unreadable 5+-way Venn intersections.
- **Module**: [`biostatistics_modules/16_venn_diagrams_upset_plots/`](biostatistics_modules/16_venn_diagrams_upset_plots/)
- **Visual Output**: `16_venn_upset_plots.png`, `venny4.png`, `upset_10_random_samples.png`

---

## Synthetic Proteomics Pipeline

All 25 modules feature a standardized companion script named **`example_proteomics.py`**, which executes the statistical method on realistic mass spectrometry data generated by `Random_Proteomics_Dataset_Generator.py`:

```bash
# 1. Generate the standardized 800-protein DIA-NN/Spectronaut benchmark table
python Random_Proteomics_Dataset_Generator.py

# 2. Run any module on the proteomics benchmark
python biostatistics_modules/17_volcano_plots/example_proteomics.py
python biostatistics_modules/05b_nipals_no_imputation_pca/example_proteomics.py
python biostatistics_modules/09_3_false_discovery_rate/example_proteomics.py
```

---

## R vs. Python Rosetta Stone Table

| Biostatistical Task | Standard R Function / Package | Production Python Function / Library |
| :--- | :--- | :--- |
| **Exploratory Data Summary** | `summary(df)`, `str(df)` | `df.describe()`, `df.info()` |
| **Missingness Matrix** | `VIM::aggr()`, `mice::md.pattern()` | `missingno.matrix()`, `missingno.heatmap()` |
| **Missing Imputation (k-NN)** | `VIM::kNN()`, `mice(method="pmm")` | `sklearn.impute.KNNImputer(n_neighbors=5)` |
| **Z-Score Normalization** | `scale(x, center=TRUE, scale=TRUE)` | `sklearn.preprocessing.StandardScaler()` |
| **Normality Test (Shapiro)** | `shapiro.test(x)` | `scipy.stats.shapiro(x)` |
| **Variance Homogeneity** | `car::leveneTest(y ~ group)` | `scipy.stats.levene(*groups, center="median")` |
| **Two-Sample t-Test (Welch)** | `t.test(x, y, var.equal=FALSE)` | `scipy.stats.ttest_ind(x, y, equal_var=False)` |
| **Mann-Whitney U-Test** | `wilcox.test(x, y, exact=FALSE)` | `scipy.stats.mannwhitneyu(x, y, alternative="two-sided")` |
| **One-Way ANOVA** | `aov(response ~ group, data=df)` | `scipy.stats.f_oneway(*groups)` |
| **Kruskal-Wallis Test** | `kruskal.test(response ~ group)` | `scipy.stats.kruskal(*groups)` |
| **Tukey HSD Post-Hoc** | `TukeyHSD(aov_fit)` | `statsmodels.stats.multicomp.pairwise_tukeyhsd()` |
| **Dunn's Post-Hoc with FDR** | `FSA::dunnTest(response ~ group, method="bh")` | `scikit_posthocs.posthoc_dunn(p_adjust="fdr_bh")` |
| **Cohen's d Effect Size** | `effsize::cohen.d(x, y)` | `numpy` pooled standard deviation calculation |
| **Benjamini-Hochberg FDR** | `p.adjust(p, method="BH")` | `statsmodels.stats.multitest.multipletests(method="fdr_bh")` |
| **Storey q-value (pFDR)** | `qvalue::qvalue(p)$qvalues` | `09_3_fdr.calculate_storey_qvalues(p)` |
| **PERMANOVA** | `vegan::adonis2(dist ~ group, permutations=999)` | `skbio.stats.distance.permanova(dm, grouping)` |
| **ANOSIM** | `vegan::anosim(dist, grouping, permutations=999)` | `skbio.stats.distance.anosim(dm, grouping)` |
| **Principal Component Analysis** | `prcomp(df, center=TRUE, scale.=TRUE)` | `sklearn.decomposition.PCA(n_components=k)` |
| **NIPALS Missing PCA** | `nipals::nipals(df, ncomp=k)` | `05b_nipals_pca.perform_nipals_pca(df)` |
| **Non-Metric MDS (NMDS)** | `vegan::metaMDS(df, distance="bray")` | `sklearn.manifold.MDS(metric=False)` |
| **Hierarchical Clustering** | `hclust(dist(df), method="ward.D2")` | `scipy.cluster.hierarchy.linkage(method="ward")` |
| **Clustered Heatmap** | `pheatmap::pheatmap(df, scale="row")` | `seaborn.clustermap(df, z_score=0, cmap="vlag")` |
| **Volcano Plot** | `EnhancedVolcano::EnhancedVolcano()` | `17_volcano_plots.plot_volcano(df)` |
| **MA Plot** | `limma::plotMA()` / `DESeq2::plotMA()` | `18_ma_plots.plot_ma(df)` |
| **Random Forest Classifier** | `randomForest::randomForest(x, y, ntree=500)` | `sklearn.ensemble.RandomForestClassifier(n_estimators=500)` |
| **Stratified K-Fold CV** | `caret::trainControl(method="cv", number=5)` | `sklearn.model_selection.StratifiedKFold(n_splits=5)` |
| **ROC-AUC Calculation** | `pROC::roc(y, prob)$auc` | `sklearn.metrics.roc_auc_score(y, prob)` |
| **Venn Diagram (2–4 Sets)** | `VennDiagram::venn.diagram()` | `matplotlib_venn.venn3()` / `venny4py` |
| **UpSet Plot ($\ge 5$ Sets)** | `UpSetR::upset(fromList(sets))` | `upsetplot.UpSet(from_contents(sets)).plot()` |

---

## Installation & Environment Setup

### Prerequisites
- Python 3.9 or higher
- Recommended: Virtual environment (`venv` or `uv`)

### Automated Setup

```bash
# 1. Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install all required biostatistical and scientific packages
pip install --upgrade pip
pip install numpy pandas scipy statsmodels scikit-learn scikit-bio scikit-posthocs matplotlib seaborn missingno upsetplot matplotlib-venn
```

Or with `uv` (lightning-fast resolver):
```bash
uv venv
source .venv/bin/activate
uv pip install numpy pandas scipy statsmodels scikit-learn scikit-bio scikit-posthocs matplotlib seaborn missingno upsetplot matplotlib-venn
```

---

## Repository Structure

```text
biostatistics_in_python/
├── README.md                                  # Central Scientific Documentation & Reference Guide
├── LICENSE                                    # MIT License
├── statistical scripts.png                    # End-to-End Workflow & Pipeline Architecture Diagram
├── Random_Proteomics_Dataset_Generator.py     # Shared DIA-NN / Spectronaut Proteomics Dataset Generator
└── biostatistics_modules/                     # 25 Modular Biostatistics Packages
    ├── 01_data_quality_assessment/            # QC, MCAR/MAR/MNAR missingness, IQR outlier detection
    │   ├── README.md                          # Theory, math, decision table, R vs Python
    │   ├── 01_data_quality_assessment.py      # Standalone executable Python script
    │   ├── 01_data_quality_assessment.ipynb   # Interactive clean Jupyter Notebook
    │   ├── example_proteomics.py              # Synthetic DIA-NN proteomics benchmark
    │   └── 01_data_quality_assessment.png     # Rendered high-resolution visualization
    ├── 01b_pairwise_replicate_scatter/        # Replicate QC, identity line (y=x), R² determination
    │   ├── README.md
    │   ├── 01b_pairwise_replicate_scatter.py
    │   ├── 01b_pairwise_replicate_scatter.ipynb
    │   ├── example_proteomics.py
    │   └── pairwise_scatter_proteomics_template.png
    ├── 02_missing_data_handling/              # Missing data imputation (k-NN, Median, MissForest)
    │   ├── README.md
    │   ├── 02_missing_data_handling.py
    │   ├── 02_missing_data_handling.ipynb
    │   ├── example_proteomics.py
    │   └── 02_missing_data_handling.png
    ├── 02_variance_homogeneity_tests/         # Levene, Bartlett, Fligner-Killeen homoscedasticity tests
    │   ├── README.md
    │   ├── 02_variance_homogeneity_tests.py
    │   ├── 02_variance_homogeneity_tests.ipynb
    │   ├── example_proteomics.py
    │   └── 02_variance_homogeneity_tests.png
    ├── 03_abundance_threshold_filtering/      # Prevalence filtering, variance thresholds, intensity filters
    │   ├── README.md
    │   ├── 03_abundance_threshold_filtering.py
    │   ├── 03_abundance_threshold_filtering.ipynb
    │   ├── example_proteomics.py
    │   └── 03_abundance_filtering.png
    ├── 04_normalization_zscore_scaling/       # Median centering, Quantile normalization, Z-Score scaling
    │   ├── README.md
    │   ├── 04_normalization_zscore_scaling.py
    │   ├── 04_normalization_zscore_scaling.ipynb
    │   ├── example_proteomics.py
    │   └── 04_normalization_and_scaling.png
    ├── 05_shapiro_wilk_normality_test/        # Shapiro-Wilk, D'Agostino-Pearson, Q-Q probability plots
    │   ├── README.md
    │   ├── 05_shapiro_wilk_normality_test.py
    │   ├── 05_shapiro_wilk_normality_test.ipynb
    │   ├── example_proteomics.py
    │   └── 05_shapiro_wilk_normality.png
    ├── 05b_nipals_no_imputation_pca/          # NIPALS No-Imputation PCA for incomplete omics matrices
    │   ├── README.md
    │   ├── 05b_nipals_no_imputation_pca.py
    │   ├── 05b_nipals_no_imputation_pca.ipynb
    │   ├── example_proteomics.py
    │   └── 05b_nipals_no_imputation_pca.png
    ├── 06_pearson_spearman_correlation/       # Pearson, Spearman rank, and Kendall correlation matrices
    │   ├── README.md
    │   ├── 06_pearson_spearman_correlation.py
    │   ├── 06_pearson_spearman_correlation.ipynb
    │   ├── example_proteomics.py
    │   └── 06_correlation_analysis.png
    ├── 07_ttest_mann_whitney_utest/           # Two-sample Student / Welch t-test & Mann-Whitney U-test
    │   ├── README.md
    │   ├── 07_ttest_mann_whitney_utest.py
    │   ├── 07_ttest_mann_whitney_utest.ipynb
    │   ├── example_proteomics.py
    │   └── 07_two_group_comparisons.png
    ├── 08_anova_kruskal_wallis_test/          # One-Way ANOVA & Kruskal-Wallis H-test
    │   ├── README.md
    │   ├── 08_anova_kruskal_wallis_test.py
    │   ├── 08_anova_kruskal_wallis_test.ipynb
    │   ├── example_proteomics.py
    │   └── 08_anova_kruskal_wallis.png
    ├── 09_1_posthoc_tests/                    # Pairwise Post-Hoc: Tukey HSD, Games-Howell, Dunn FDR
    │   ├── README.md
    │   ├── 09_1_posthoc_tests.py
    │   ├── 09_1_posthoc_tests.ipynb
    │   ├── example_proteomics.py
    │   └── 09_1_posthoc_tests.png
    ├── 09_2_effect_size/                      # Effect sizes: Cohen's d, Hedges' g, Eta-Squared (η²)
    │   ├── README.md
    │   ├── 09_2_effect_size.py
    │   ├── 09_2_effect_size.ipynb
    │   ├── example_proteomics.py
    │   └── 09_2_effect_size_analysis.png
    ├── 09_3_false_discovery_rate/             # Postdoc Multiple Testing: FWER, FDR, Storey q, locFDR, IHW
    │   ├── README.md
    │   ├── 09_3_false_discovery_rate.py
    │   ├── 09_3_false_discovery_rate.ipynb
    │   ├── example_proteomics.py
    │   └── 09_3_false_discovery_rate.png
    ├── 09_anosim_permanova/                   # Multivariate distance permutations: PERMANOVA & ANOSIM
    │   ├── README.md
    │   ├── 09_anosim_permanova.py
    │   ├── 09_anosim_permanova.ipynb
    │   ├── example_proteomics.py
    │   └── 09_multivariate_anosim_permanova.png
    ├── 10_kmeans_hierarchical_dbscan_clustering/ # K-Means, Hierarchical (Ward), DBSCAN & Silhouette
    │   ├── README.md
    │   ├── 10_kmeans_hierarchical_dbscan_clustering.py
    │   ├── 10_kmeans_hierarchical_dbscan_clustering.ipynb
    │   ├── example_proteomics.py
    │   └── 10_unsupervised_clustering.png
    ├── 11_pca_pcoa_nmds_ordination/           # PCA, PCoA (Bray-Curtis/Euclidean), NMDS & Scree analysis
    │   ├── README.md
    │   ├── 11_pca_pcoa_nmds_ordination.py
    │   ├── 11_pca_pcoa_nmds_ordination.ipynb
    │   ├── example_proteomics.py
    │   └── 11_multivariate_ordination.png
    ├── 12_decision_trees_neural_networks/     # Decision Trees & Multi-Layer Perceptrons (MLP)
    │   ├── README.md
    │   ├── 12_decision_trees_neural_networks.py
    │   ├── 12_decision_trees_neural_networks.ipynb
    │   ├── example_proteomics.py
    │   └── 12_decision_trees_and_neural_nets.png
    ├── 13_cross_validation/                   # Stratified K-Fold, Repeated K-Fold, and LOOCV validation
    │   ├── README.md
    │   ├── 13_cross_validation.py
    │   ├── 13_cross_validation.ipynb
    │   ├── example_proteomics.py
    │   └── 13_cross_validation.png
    ├── 14_model_evaluation_metrics/           # ROC-AUC, Precision-Recall (PR-AUC), Confusion Matrices
    │   ├── README.md
    │   ├── 14_model_evaluation_metrics.py
    │   ├── 14_model_evaluation_metrics.ipynb
    │   ├── example_proteomics.py
    │   └── 14_model_evaluation_metrics.png
    ├── 15_random_forests/                     # Random Forest ensembles & Gini feature importance
    │   ├── README.md
    │   ├── 15_random_forests.py
    │   ├── 15_random_forests.ipynb
    │   ├── example_proteomics.py
    │   └── 15_random_forests.png
    ├── 16_venn_diagrams_upset_plots/          # 2- to 5-Way Venn diagrams, UpSet plots, region tables
    │   ├── README.md
    │   ├── 16_venn_diagrams_upset_plots.py
    │   ├── 16_venn_diagrams_upset_plots.ipynb
    │   ├── example_proteomics.py
    │   ├── 16_venn_upset_plots.png
    │   ├── venny4.png
    │   └── upset_10_random_samples.png
    ├── 17_volcano_plots/                      # Differential expression Volcano plots (log2FC vs -log10 padj)
    │   ├── README.md
    │   ├── 17_volcano_plots.py
    │   ├── 17_volcano_plots.ipynb
    │   ├── example_proteomics.py
    │   └── 17_volcano_plots.png
    ├── 18_ma_plots/                           # MA plots (ratio-intensity diagnostic bias curves)
    │   ├── README.md
    │   ├── 18_ma_plots.py
    │   ├── 18_ma_plots.ipynb
    │   ├── example_proteomics.py
    │   └── 18_ma_plots.png
    └── 19_clustered_heatmaps/                 # Hierarchical clustered heatmaps with condition annotation bars
        ├── README.md
        ├── 19_clustered_heatmaps.py
        ├── 19_clustered_heatmaps.ipynb
        ├── example_proteomics.py
        └── 19_clustered_heatmaps.png
```

---

## License

Distributed under the **[MIT License](LICENSE)**. Open source and freely usable for academic and commercial scientific research.
