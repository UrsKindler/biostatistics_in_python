# T-Test & Mann-Whitney U-Test

## Overview
- **Purpose**: Compare means/distributions between two independent or paired groups
- **Use Case**: Treatment vs control, before vs after measurements
- **Prerequisites**: Independent observations, continuous or ordinal data

## When to Use
### Decision Criteria
- ✅ **Use T-Test when**: Data is normally distributed (Shapiro-Wilk p > 0.05)
- ✅ **Use Mann-Whitney when**: Data is not normally distributed or ordinal
- ✅ **Use Paired versions when**: Same subjects measured twice
- ❌ **Don't use when**: More than two groups (use ANOVA instead)

## R Legacy vs Python Modern

| Aspect | R (Legacy) | Python (Modern) |
|--------|------------|-----------------|
| Package | `base R` | `scipy.stats` |
| T-Test Function | `t.test(group1, group2, paired=FALSE)` | `ttest_ind(group1, group2)` |
| U-Test Function | `wilcox.test(group1, group2)` | `mannwhitneyu(group1, group2)` |
| Normality Test | `shapiro.test(data)` | `shapiro(data)` |
| P-adjustment | `p.adjust(p, method="BH")` | `multipletests(p, method='fdr_bh')` |
| Advantages | Built-in corrections | Better ecosystem integration |
| Disadvantages | Limited ML pipeline | Requires multiple imports |

## Quick Start Code


