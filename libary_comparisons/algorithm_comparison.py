import numpy as np
from scipy.stats import friedmanchisquare, wilcoxon
from itertools import combinations
from statsmodels.stats.multitest import multipletests

data = np.array([
    [0.81, 0.76, 0.89],
    [0.83, 0.78, 0.88],
    [0.79, 0.77, 0.87],
    [0.84, 0.75, 0.90],
    [0.80, 0.79, 0.91]
])

# Friedman test
stat, p_friedman = friedmanchisquare(*[data[:, i] for i in range(3)])
print("Friedman test statistic:", stat)
print("Friedman test p-value:", p_friedman)

# Pairwise Wilcoxon
if p_friedman < 0.05:
    print("\nFriedman significant → Running Wilcoxon pairwise tests...\n")

pairs = list(combinations(range(3), 2))
pvals = []

for i, j in pairs:
    _, p = wilcoxon(data[:, i], data[:, j])
    pvals.append(p)

print("Pairwise Wilcoxon p-values:", pvals)

# Holm correction
rej, pvals_corrected, _, _ = multipletests(pvals, method='holm')
print("\nHolm-adjusted p-values:", pvals_corrected)
print("Rejected hypotheses:", rej)



