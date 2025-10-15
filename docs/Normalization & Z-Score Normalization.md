## **4. Normalization & Z-Score Normalization**

### Overview
- **Purpose**: Scale data to comparable ranges and normalize distributions
- **Use Case**: Before PCA, clustering, or when features have different scales
- **Prerequisites**: Numeric data, understanding of data distribution

### When to Use
#### Decision Criteria
- **Use Z-score when**: Features have different scales, before PCA/clustering
- **Use Min-Max when**: Need bounded values (0-1), neural networks
- **Use Log transform when**: Right-skewed data, fold-change data
- **Use Robust scaling when**: Outliers present

### R Legacy vs Python Modern

| Aspect | R | Python |
|--------|---|--------|
| Package | `base R`, `scale` | `sklearn.preprocessing`, `scipy` |
| Z-score | `scale(data)` | `StandardScaler()` |
| Min-Max | `(data - min) / (max - min)` | `MinMaxScaler()` |
| Log Transform | `log(data + 1)` | `np.log1p(data)` |
| Robust Scaling | Custom with median/MAD | `RobustScaler()` |

### Quick Start Code

**R Example:**
```r
# Z-score normalization (base R)
data_scaled <- scale(data)

# Min-max normalization
normalize <- function(x) {
  return((x - min(x)) / (max(x) - min(x)))
}
data_normalized <- apply(data, 2, normalize)

# Log transformation
data_log <- log(data + 1)
```

**Python Simple:**
```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import numpy as np

# Z-score normalization
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df), 
                        columns=df.columns, index=df.index)

# Min-max normalization
minmax_scaler = MinMaxScaler()
df_minmax = pd.DataFrame(minmax_scaler.fit_transform(df), 
                        columns=df.columns, index=df.index)

# Log transformation
df_log = np.log1p(df)  # log(1 + x) to handle zeros

# Robust scaling (less sensitive to outliers)
robust_scaler = RobustScaler()
df_robust = pd.DataFrame(robust_scaler.fit_transform(df), 
                        columns=df.columns, index=df.index)
```
