# State of the Art: Multiple Testing Correction in Omics & High-Throughput Biostatistics

> **Target Level**: Postdoctoral / Principal Investigator Biostatistics Reference  
> **Module Implementation**: [`biostatistics_modules/09_3_false_discovery_rate/`](biostatistics_modules/09_3_false_discovery_rate/)

---

## 1. Mathematical Framework & The Multiplicity Problem

When conducting $m$ simultaneous hypothesis tests:
$$H_{01}, H_{02}, \dots, H_{0m} \quad \text{vs.} \quad H_{11}, H_{12}, \dots, H_{1m}$$
with $m \sim 10^3 - 10^6$ features (e.g. mass-spectrometry proteomics, RNA-seq, single-cell multi-omics, GWAS), evaluating each hypothesis at nominal $\alpha = 0.05$ yields an expected number of false discoveries equal to $\alpha \cdot m_0 \approx 0.05 \cdot m$.

### Hypothesis Outcome Matrix (Benjamini & Hochberg, 1995)

| Hypothesis State | Declared Non-Significant ($H_0$ Not Rejected) | Declared Significant ($H_0$ Rejected) | Total |
| :--- | :---: | :---: | :---: |
| **True Null ($H_0$ True)** | $U$ | $V$ *(Type I Errors / False Positives)* | $m_0$ |
| **Non-Null ($H_0$ False)** | $T$ *(Type II Errors / False Negatives)* | $S$ *(True Positives)* | $m_1$ |
| **Total** | $m - R$ | $R$ *(Total Discoveries)* | $m$ |

---

## 2. Error Rate Metrics & Guarantees

### A. Family-Wise Error Rate (FWER)
Probability of committing at least one Type I error across the entire collection of tests:
$$\text{FWER} = \mathbb{P}(V \ge 1) \le \alpha$$

1. **Bonferroni (1936)**: Single-step union bound correction:
   $$\tilde{p}_i = \min(1, m \cdot p_i)$$
   - *Guarantee*: Valid under arbitrary dependence structures.
   - *Limitation*: Overly conservative when tests are positively correlated, resulting in severe statistical power loss at large $m$.
2. **Holm-Bonferroni (1979)**: Step-down sequential procedure on ordered $p$-values $p_{(1)} \le p_{(2)} \le \dots \le p_{(m)}$:
   $$\tilde{p}_{(i)} = \min\left(1, \max_{j \le i} (m - j + 1) p_{(j)}\right)$$
   - *Guarantee*: Uniformly more powerful than Bonferroni without any independence assumptions.
3. **Hochberg (1988) & Hommel (1988)**: Step-up sequential procedure valid under Simes\' inequality (Positive Regression Dependency on Subsets - PRDS):
   $$\tilde{p}_{(i)} = \min_{j \ge i} (m - j + 1) p_{(j)}$$
4. **Westfall-Young (1993) Permutation Step-Down ($\text{max}T / \text{min}P$)**:
   - Preserves arbitrary empirical correlation structures by permuting phenotype/condition labels across all $m$ features simultaneously.
   - Gold standard for exact FWER control when sample sizes permit sufficient permutation depth ($B \ge 1,000$).

---

### B. False Discovery Rate (FDR)
Expected proportion of false rejections among all rejected hypotheses:
$$\text{FDR} = \mathbb{E}[Q], \quad \text{where } Q = \begin{cases} \frac{V}{R} & \text{if } R > 0 \\ 0 & \text{if } R = 0 \end{cases}$$

1. **Benjamini-Hochberg (BH, 1995)**: Step-up procedure:
   $$k = \max \left\{ i : p_{(i)} \le \frac{i}{m} \alpha \right\}, \quad \text{Reject } H_{(1)}, \dots, H_{(k)}$$
   - *Guarantee*: Controls $\text{FDR} \le \frac{m_0}{m}\alpha \le \alpha$ under PRDS (e.g., multivariate Gaussian with positive correlations).
2. **Benjamini-Yekutieli (BY, 2001)**: Controls FDR under arbitrary / negative dependence:
   $$\tilde{p}_{(i)} \le \frac{i}{m \cdot c(m)} \alpha, \quad \text{where } c(m) = \sum_{i=1}^m \frac{1}{i} \approx \ln(m) + \gamma + \frac{1}{2m}$$

---

### C. Empirical Bayes, Positive FDR & Storey\'s q-value (Storey, 2002, 2003)

Under the mixture model, $p$-values are a mixture of Uniform(0,1) true nulls and non-nulls skewed towards 0:
$$f(p) = \pi_0 \cdot 1 + (1 - \pi_0) \cdot f_1(p)$$

1. **Estimating the True Null Proportion $\pi_0 = \frac{m_0}{m}$**:
   $$\hat{\pi}_0(\lambda) = \frac{\#\{p_i > \lambda\}}{m(1 - \lambda)}$$
   Evaluated across $\lambda \in [0.05, 0.95]$ and smoothed via natural cubic splines or bootstrap to extrapolate $\hat{\pi}_0(1)$.
2. **The $q$-value**:
   The minimum FDR at which test $i$ is declared significant:
   $$q(p_{(i)}) = \min_{j \ge i} \left( \frac{\hat{\pi}_0 \cdot m \cdot p_{(j)}}{j} \right)$$
   - *Advantage*: Increases discovery power by factor $\frac{1}{\hat{\pi}_0}$ compared to standard BH (crucial when $m_1$ is large, e.g. 20-50% of the proteome).

---

### D. Local False Discovery Rate ($\text{locFDR}$, Efron 2004, 2008)

While tail-area FDR assesses $\mathbb{P}(\text{Null} \mid Z \in \text{Rejection Region})$, **Local FDR** quantifies the posterior probability of a specific test statistic $z$ being a false discovery:
$$\text{locFDR}(z) = \mathbb{P}(H_0 \mid Z = z) = \frac{\pi_0 f_0(z)}{f(z)}$$
where $f_0(z) \sim \mathcal{N}(0, 1)$ (theoretical null) or $\mathcal{N}(\mu_{\text{emp}}, \sigma^2_{\text{emp}})$ (empirical null estimating unmodeled batch effects/correlation).

---

### E. Covariate-Modulated FDR: Independent Hypothesis Weighting (IHW)
Ignatiadis et al. (*Nature Methods*, 2016) demonstrated that testing power is enhanced without inflating FDR by assigning non-negative weights $w_i \ge 0$ ($\sum w_i = m$) based on an independent informative covariate $X_i$ (e.g., mean protein abundance, peptide count, genomic conservation):
$$p_i \le w_i \cdot \alpha \frac{k}{m}$$

---

### F. High-Dimensional Model-X Knockoffs (Candès et al., 2018)
Exact FDR control in high-dimensional feature selection without distributional assumptions on $Y \mid X$. Constructs synthetic knockoff features $\tilde{X}$ that mimic the correlation structure of $X$ but are conditionally independent of $Y$, rejecting features whose test statistic substantially exceeds their knockoff counterpart.

---

## 3. Proteomics & Mass Spectrometry Specific FDR Concepts

In quantitative bottom-up proteomics (DDA / DIA-NN / Spectronaut / MaxQuant):
1. **Target-Decoy Competition (TDC)**:
   $$\text{FDR}_{\text{PSM}} \approx \frac{N_{\text{decoy}}}{N_{\text{target}}}$$
2. **Picked Protein Group FDR (Savitski et al., 2015)**:
   Resolves the protein inference problem where shared peptides create dependencies. Competes target and decoy protein groups directly to prevent FDR inflation.
3. **Hierarchical FDR**:
   PSM-level FDR $\neq$ Peptide-level FDR $\neq$ Protein-level FDR. Strict multi-level FDR filtering prevents accumulation of false positive identifications at the protein tier.

---

## 4. Decision Matrix: Which Correction Method to Choose?

| Scenario | Target Metric | Recommended Procedure | Tool / Implementation |
| :--- | :--- | :--- | :--- |
| **Strict Single Biomarker Validation** | FWER $\le 0.05$ | Holm-Bonferroni / Westfall-Young | `statsmodels.stats.multitest.multipletests(method="holm")` |
| **Standard Proteomics / RNA-seq Screening** | FDR $\le 0.05$ | Benjamini-Hochberg (BH) | `multipletests(method="fdr_bh")` |
| **High Signal Density ($m_1 \gg 0$)** | pFDR / $q$-value | Storey's $q$-value ($\hat{\pi}_0$ Spline) | `09_3_false_discovery_rate.calculate_storey_qvalues()` |
| **Complex Covariates Available (e.g. Abundance)** | Weighted FDR | Independent Hypothesis Weighting (IHW) | `IHW` (R) / Stratified BH (Python) |
| **Negative / Unknown Dependencies** | FDR $\le 0.05$ | Benjamini-Yekutieli (BY) | `multipletests(method="fdr_by")` |
| **Pointwise Biomarker Confidence** | Posterior $P(H_0 \mid z)$ | Local FDR ($\text{locFDR}$) | `scipy.stats` mixture / `statsmodels` |

---

## 5. Implementation in Python

All methods are implemented with clean scripts, Jupyter Notebooks, and synthetic DIA-NN proteomics examples in:
👉 [`biostatistics_modules/09_3_false_discovery_rate/`](biostatistics_modules/09_3_false_discovery_rate/)
