**1. Data Quality Assessment**

  ### Overview
- **Purpose**: Evaluate data completeness, identify outliers, and assess data integrity before statistical analysis
- **Use Case**: Initial data exploration, missing data pattern identification, outlier detection
- **Prerequisites**: Raw dataset with mixed data types

### When to Use
#### Decision Criteria
- **Use always**: As first step in any data analysis workflow
- **Essential when**: High-dimensional data, clinical/biological datasets with expected missing patterns
- **Critical for**: Datasets from multiple sources or collection timepoints
- **Don't skip when**: Working with real-world messy data

### R vs Python 

| Aspect | R | Python |
|--------|---|--------|
| Package | `VIM`, `mice`, `naniar` | `missingno`, `pandas` |
| Missing Pattern | `VIM::aggr(data)` | `missingno.matrix(df)` |
| Data Summary | `summary(data)` | `df.describe()` |
| Outlier Detection | `boxplot.stats(data)$out` | `scipy.stats.zscore()` |
| Data Structure | `str(data)` | `df.info()` |

### Quick Start Code

**R Example:**
```r
# Load required libraries
library(VIM)

# Basic data inspection
summary(data)
str(data)

# Missing data visualization  
aggr(data, col=c('navyblue','red'), numbers=TRUE, sortVars=TRUE)

# Outlier detection
outliers <- boxplot.stats(data$variable)$out
```

**Python Simple:**
```python
import pandas as pd
import numpy as np
import missingno as msno

# Load and inspect
df = pd.read_csv('data.csv', index_col=0)
print(df.info())
print(df.describe())

# Missing data visualization
msno.matrix(df)

# Simple outlier detection
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
outliers = ((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).sum()
print(f"Outliers per column:\n{outliers}")
```
