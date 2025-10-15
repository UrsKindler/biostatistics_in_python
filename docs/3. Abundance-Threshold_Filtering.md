## **3. Abundance-Threshold Filtering**

### Overview
- **Purpose**: Remove features with low abundance/prevalence to reduce noise and multiple testing burden
- **Use Case**: High-dimensional omics data, microbiome analysis, proteomics
- **Prerequisites**: Count or abundance data with many zero/low values

### When to Use
#### Decision Criteria
- **Use always with**: Microbiome, proteomics, metabolomics data
- **Use when**: >1000 features with sparse data
- **Threshold typically**: 0.1% relative abundance, present in >10% samples
- **Don't use when**: Already filtered data, small feature sets

### R vs Python 

| Aspect | R | Python |
|--------|---|--------|
| Package | `edgeR`, `DESeq2`, custom | `pandas`, `sklearn.feature_selection` |
| Relative Abundance | `sweep(data, 2, colSums(data), "/")` | `df.div(df.sum(axis=1), axis=0)` |
| Prevalence Filter | `rowSums(data > 0) >= threshold` | `(df > 0).sum() >= threshold` |
| Variance Filter | Custom implementation | `VarianceThreshold()` |
| CPM Filter | `edgeR::cpm(data)` | Custom calculation |

### Quick Start Code

**R Example:**
```r
# Load required libraries
library(edgeR)  # if using CPM filtering

# Prevalence filtering
prevalence_threshold <- 0.1
prevalence <- rowSums(data > 0) / ncol(data)
data_filtered <- data[prevalence >= prevalence_threshold, ]

# Relative abundance filtering
rel_abundance <- sweep(data, 2, colSums(data), "/")
abundance_threshold <- 0.001
keep_features <- rowSums(rel_abundance > abundance_threshold) >= 3
data_filtered <- data[keep_features, ]
```

**Python Simple:**
```python
# Prevalence filtering (samples as rows, features as columns)
prevalence_threshold = 0.1
prevalence = (df > 0).sum() / len(df)
df_filtered = df.loc[:, prevalence >= prevalence_threshold]

# Relative abundance filtering
rel_abundance = df.div(df.sum(axis=1), axis=0)
abundance_threshold = 0.001
min_samples = 3
keep_features = (rel_abundance > abundance_threshold).sum() >= min_samples
df_filtered = df.loc[:, keep_features]

# Variance filtering
from sklearn.feature_selection import VarianceThreshold
selector = VarianceThreshold(threshold=0.01)
df_variance_filtered = pd.DataFrame(selector.fit_transform(df), 
                                   columns=df.columns[selector.get_support()],
                                   index=df.index)
```
