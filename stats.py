import os

import numpy as np
from scipy import stats

basedir = 'd6'
dirs = ['one','two','three','four','five','six']
fulldirs = []

for dir in dirs:
    fulldirs.append(os.path.join(basedir,dir))

vals = []
for dir in fulldirs:
    if os.path.isfile(dir):
        continue
    
    files = os.listdir(dir)
    
    val = len(files)
    vals.append(val)
    print(f'{dir:<5} -> {val}')

observed_counts = vals

n_rolls = np.sum(observed_counts)

expected_counts = np.full(6, n_rolls / 6)

chi2_stat, p_value = stats.chisquare(f_obs=observed_counts, f_exp=expected_counts)

# Calculate percentage deviation
deviation = ((observed_counts - expected_counts) / expected_counts) * 100

print(f"Total Rolls: {n_rolls}")
print(f"Observed:    {observed_counts}")
print(f"Expected:    {expected_counts.round(2)}")
print(f"Deviations:  {deviation.round(2)} %")
print("-" * 30)
print(f"Chi2 Stat:   {chi2_stat:.4f}")
print(f"P-Value:     {p_value:.4f}")