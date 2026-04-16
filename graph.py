import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
from scipy.stats import chisquare

# 1. Load data
# Replace 'data.csv' with your actual filename

rolls = pd.concat([pd.read_csv(f)
                  for f in sorted(glob.glob("*.csv"))])['ROLL_VALUE'].tolist()
# rolls = pd.read_csv('output.txt')['ROLL_VALUE'].tolist()
# Determine unique outcomes (e.g., 1, 2, 3, 4, 5, 6)
unique_vals = sorted(np.unique(rolls))
k = len(unique_vals)
n_total = len(rolls)

# Tracking arrays
observed_counts = np.zeros((n_total, k))
expected_counts = np.zeros(n_total)
chi_sq_stats = np.zeros(n_total)
p_values = np.zeros(n_total)

# Mapping values to array indices
val_to_idx = {val: i for i, val in enumerate(unique_vals)}
current_counts = np.zeros(k)

# 2. Iterative Calculation
for i in range(n_total):
    val = rolls[i]
    current_counts[val_to_idx[val]] += 1
    observed_counts[i] = current_counts

    n = i + 1
    expected = n / k
    expected_counts[i] = expected

    # Calculate Chi-Square and P-Value
    # Null hypothesis: The data is uniformly distributed (Expected = n/k for all)
    f_exp = np.full(k, expected)
    stat, p = chisquare(current_counts, f_exp=f_exp)

    chi_sq_stats[i] = stat
    p_values[i] = p

# 3. Visualization
fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
x_range = range(1, n_total + 1)


# --- Graph 1: Actual vs Expected Frequencies ---
for idx, val in enumerate(unique_vals):
    axes[0].plot(x_range, observed_counts[:, idx], label=f'Value: {val}')
axes[0].plot(x_range, expected_counts, color='black',
             linestyle='--', linewidth=2, label='Expected (n/k)')
axes[0].set_title('Cumulative Frequencies vs. Expected')
axes[0].set_ylabel('Frequency')
axes[0].legend(loc='upper left', fontsize='small', ncol=2)
axes[0].grid(True, alpha=0.3)

# --- Graph 2: Combined Chi-Square and P-Value ---
color_chi = 'tab:red'
axes[1].set_title('$\chi^2$ and P-Value')
axes[1].set_xlabel('Number of Rolls (n)')
axes[1].set_ylabel('$\chi^2$ Value', color=color_chi)
axes[1].plot(x_range, chi_sq_stats, color=color_chi, label='$\chi^2$ Stat')
axes[1].tick_params(axis='y', labelcolor=color_chi)
axes[1].grid(True, alpha=0.3)

# Create the twin axis for P-Value
ax_p = axes[1].twinx()
color_p = 'tab:green'
ax_p.set_ylabel('P-Value', color=color_p)
ax_p.plot(x_range, p_values, color=color_p, label='P-Value')
ax_p.tick_params(axis='y', labelcolor=color_p)

# Add the significance threshold
ax_p.axhline(y=0.05, color='orange', linestyle='--',
             alpha=0.6, label='Alpha (0.05)')
ax_p.set_ylim(0, 1.05)

# Merge legends from the twin axes
lines_1, labels_1 = axes[1].get_legend_handles_labels()
lines_p, labels_p = ax_p.get_legend_handles_labels()
ax_p.legend(lines_1 + lines_p, labels_1 + labels_p, loc='center right')


# 4. Final Summary Analysis
# Extracting the last values from our cumulative arrays
n_rolls = n_total
final_observed = observed_counts[-1]
final_expected = expected_counts[-1]
# 4. Final Summary Analysis
n_rolls = n_total
final_observed = observed_counts[-1]
final_expected = expected_counts[-1]
final_chi2 = chi_sq_stats[-1]
final_p = p_values[-1]

deviations_pct = ((final_observed - final_expected) / final_expected) * 100

print(f"\nTotal Rolls: {n_rolls}")
print(f"Observed:    {final_observed}")
print(f"Expected:    {final_expected:.2f}")
print(f"Deviations:  {np.round(deviations_pct, 2)} %")
print("-" * 30)

# Identify over and under represented values
over_represented = [str(unique_vals[i])
                    for i, d in enumerate(deviations_pct) if d > 0]
under_represented = [str(unique_vals[i])
                     for i, d in enumerate(deviations_pct) if d < 0]
balanced = [str(unique_vals[i])
            for i, d in enumerate(deviations_pct) if d == 0]

print(
    f"Over-represented (+):  {', '.join(over_represented) if over_represented else 'None'}")
print(
    f"Under-represented (-): {', '.join(under_represented) if under_represented else 'None'}")
if balanced:
    print(f"Perfectly balanced:    {', '.join(balanced)}")

print("-" * 30)
print(f"Chi2 Stat:   {final_chi2:.4f}")
print(f"P-Value:     {final_p:.4f}")

# ... (rest of your assessment logic)
if final_p > 0.90:
    assessment = "EXTREMELY FAIR (Strongly consistent with uniform distribution)"
elif final_p > 0.10:
    assessment = "FAIR (Likely random variation)"
elif final_p > 0.05:
    assessment = "UNCERTAIN (Weak evidence of bias, but usually considered fair)"
elif final_p > 0.01:
    assessment = "LIKELY BIASED (Statistically significant deviation)"
else:
    assessment = "HEAVILY BIASED (Highly significant evidence of loading)"

print(f"Conclusion: {assessment}")
print("-" * 30)

plt.tight_layout()
plt.show()
