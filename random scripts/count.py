import os

import numpy as np
from scipy import stats

dirs = ['d6/one', 'd6/two', 'd6/three', 'd6/four', 'd6/five', 'd6/six']
# dirs = os.listdir()
vals = []
for dir in dirs:
    if os.path.isfile(dir):
        continue

    files = os.listdir(dir)

    val = len(files)
    vals.append(val)
    print(f'{dir:<5} -> {val}')

observed_counts = vals

n_rolls = np.sum(observed_counts)

expected_counts = np.full(6, n_rolls / 6)

chi2_stat, p_value = stats.chisquare(
    f_obs=observed_counts, f_exp=expected_counts)

# Calculate percentage deviation
deviation = ((observed_counts - expected_counts) / expected_counts) * 100

print(f"Total Rolls: {n_rolls}")
print(f"Observed:    {observed_counts}")
print(f"Expected:    {expected_counts.round(2)}")
print(f"Deviations:  {deviation.round(2)} %")
print("-" * 30)
print(f"Chi2 Stat:   {chi2_stat:.4f}")
print(f"P-Value:     {p_value:.4f}")


if p_value > 0.90:
    assessment = "EXTREMELY FAIR (Strongly consistent with uniform distribution)"
elif p_value > 0.10:
    assessment = "FAIR (Likely random variation)"
elif p_value > 0.05:
    assessment = "UNCERTAIN (Weak evidence of bias, but usually considered fair)"
elif p_value > 0.01:
    assessment = "LIKELY BIASED (Statistically significant deviation)"
else:
    assessment = "HEAVILY BIASED (Highly significant evidence of loading)"

print(f"Conclusion: {assessment}")
print("-" * 30)
