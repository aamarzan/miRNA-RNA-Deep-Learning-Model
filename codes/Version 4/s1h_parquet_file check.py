import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chisquare

# === CONFIG ===
folder_path = r"E:\1. Github\1. miRNA-RNA-Deep-Learning-Model\dataset\prepared_dataset"
sample_rows_for_dist = 500_000  # limit for heavy groupby ops (None = all rows)

# --- Auto-detect the latest parquet file ---
parquet_files = [f for f in os.listdir(folder_path) if f.endswith(".parquet")]
if not parquet_files:
    raise FileNotFoundError(f"No parquet files found in {folder_path}")

parquet_files.sort(key=lambda f: os.path.getmtime(os.path.join(folder_path, f)), reverse=True)
file_path = os.path.join(folder_path, parquet_files[0])

print(f"📂 Analyzing file: {file_path}")

# --- Load key columns ---
cols_to_load = ["primary_id", "target_id", "competitor_id", "affinity"]
df = pd.read_parquet(file_path, columns=cols_to_load)

# --- Basic counts ---
total_rows = len(df)
unique_mirnas = df["primary_id"].nunique()
unique_targets = df["target_id"].nunique()
unique_comps = df["competitor_id"].nunique()

print("\n📊 Dataset Overview")
print(f"Total rows              : {total_rows:,}")
print(f"Unique miRNAs           : {unique_mirnas:,}")
print(f"Unique targets          : {unique_targets:,}")
print(f"Unique competitors      : {unique_comps:,}")

# --- Affinity stats ---
min_val = df["affinity"].min()
max_val = df["affinity"].max()
mean_val = df["affinity"].mean()
quartiles = df["affinity"].quantile([0.25, 0.5, 0.75])

print("\n📈 Affinity Distribution")
print(f"Lowest value   : {min_val:.4f}")
print(f"Highest value  : {max_val:.4f}")
print(f"Average (mean) : {mean_val:.4f}")
print(f"25% quartile   : {quartiles.loc[0.25]:.4f}")
print(f"50% quartile   : {quartiles.loc[0.50]:.4f}  (Median)")
print(f"75% quartile   : {quartiles.loc[0.75]:.4f}")

# --- Optional: sample for heavier analysis ---
df_sample = df if (sample_rows_for_dist is None or len(df) <= sample_rows_for_dist) else df.sample(sample_rows_for_dist, random_state=42)

# --- Frequency counts ---
mirna_counts = df_sample["primary_id"].value_counts()
target_counts = df_sample["target_id"].value_counts()
comp_counts_all = df_sample["competitor_id"].value_counts()

# Separate NO_COMPETITOR
no_comp_count = comp_counts_all.get("NO_COMPETITOR", 0)
comp_counts_real = comp_counts_all.drop(labels=["NO_COMPETITOR"], errors="ignore")

print("\n🔍 Representation Checks (sampled if large)")
print(f"miRNA frequency range   : {mirna_counts.min()/len(df_sample)*100:.3f}% – {mirna_counts.max()/len(df_sample)*100:.3f}%")
print(f"Target frequency range  : {target_counts.min()/len(df_sample)*100:.3f}% – {target_counts.max()/len(df_sample)*100:.3f}%")
if len(comp_counts_real) > 0:
    print(f"Competitor freq. range  : {comp_counts_real.min()/len(df_sample)*100:.3f}% – {comp_counts_real.max()/len(df_sample)*100:.3f}% (excluding NO_COMPETITOR)")
print(f"NO_COMPETITOR share     : {no_comp_count/len(df_sample)*100:.2f}%")

# --- Chi-square tests for uniformity ---
def chi_square_uniform(counts, label):
    counts = counts.values  # ensure array
    total = counts.sum()
    expected = [total / len(counts)] * len(counts)  # exact sum match
    chi2, p = chisquare(counts, f_exp=expected)
    rel_dev = (counts.max() - counts.min()) / (total / len(counts)) * 100
    print(f"{label} chi-square p-value: {p:.4f} | Max rel. deviation from mean: {rel_dev:.2f}% "
          f"({'Fail to reject uniformity' if p > 0.05 else 'Significant deviation'})")

print("\n📊 Chi-square Uniformity Tests")
chi_square_uniform(mirna_counts, "miRNA")
chi_square_uniform(target_counts, "Target")
if len(comp_counts_real) > 0:
    chi_square_uniform(comp_counts_real, "Competitor (excluding NO_COMPETITOR)")

# --- miRNA–NO_COMPETITOR and min/max target/competitor counts ---
no_comp_presence = df.groupby("primary_id")["competitor_id"].apply(lambda x: "NO_COMPETITOR" in set(x))
missing_no_comp = no_comp_presence[~no_comp_presence].index.tolist()

target_counts_per_mirna = df.groupby("primary_id")["target_id"].nunique()
comp_counts_per_mirna = df[df["competitor_id"] != "NO_COMPETITOR"].groupby("primary_id")["competitor_id"].nunique()

print("\n🧩 miRNA–NO_COMPETITOR & Coverage Checks")
print(f"MiRNAs missing NO_COMPETITOR: {len(missing_no_comp)}")
print(f"Targets per miRNA: min={target_counts_per_mirna.min()}, max={target_counts_per_mirna.max()}")
print(f"Competitors per miRNA (excl. NO_COMPETITOR): min={comp_counts_per_mirna.min()}, max={comp_counts_per_mirna.max()}")

# --- Target–Competitor coverage checks ---
target_has_no_comp = df.groupby("target_id")["competitor_id"].apply(lambda x: "NO_COMPETITOR" in set(x))
target_has_real_comp = df.groupby("target_id")["competitor_id"].apply(lambda x: any(c != "NO_COMPETITOR" for c in set(x)))

targets_missing_no_comp = target_has_no_comp[~target_has_no_comp].index.tolist()
targets_missing_real_comp = target_has_real_comp[~target_has_real_comp].index.tolist()

print("\n🎯 Target–Competitor Coverage Checks")
print(f"Targets missing NO_COMPETITOR: {len(targets_missing_no_comp)}")
print(f"Targets missing real competitor: {len(targets_missing_real_comp)}")
if targets_missing_no_comp:
    print(f"Example targets missing NO_COMPETITOR: {targets_missing_no_comp[:10]}")
if targets_missing_real_comp:
    print(f"Example targets missing real competitor: {targets_missing_real_comp[:10]}")

# --- Plot histograms ---
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
df_sample["affinity"].hist(bins=30, color='skyblue', edgecolor='black')
plt.title("Affinity Distribution")
plt.xlabel("Affinity")
plt.ylabel("Count")

plt.subplot(2, 2, 2)
(mirna_counts/len(df_sample)*100).hist(bins=30, color='lightgreen', edgecolor='black')
plt.title("miRNA Frequency (%)")
plt.xlabel("Frequency %")
plt.ylabel("Count")

plt.subplot(2, 2, 3)
(target_counts/len(df_sample)*100).hist(bins=30, color='salmon', edgecolor='black')
plt.title("Target Frequency (%)")
plt.xlabel("Frequency %")
plt.ylabel("Count")

plt.subplot(2, 2, 4)
if len(comp_counts_real) > 0:
    (comp_counts_real/len(df_sample)*100).hist(bins=30, color='orange', edgecolor='black')
    plt.title("Competitor Frequency (%) (Excl. NO_COMPETITOR)")
    plt.xlabel("Frequency %")
    plt.ylabel("Count")

plt.tight_layout()
plt.show()

print("\n✅ Analysis complete with visual, statistical, and coverage checks.")
