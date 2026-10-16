"""
CCA between six functional-network gradient measures and SRS behavioral scores.

Input: ./data/ABIDE2_ASD_7net.csv
Output: ./figure/scatter3.png and ./figure/region11.png
Dependencies: numpy, pandas, scikit-learn, matplotlib, seaborn

The script removes rows with missing selected variables, standardizes brain and
behavioral variables separately, fits one-component CCA, runs a permutation
test, prints the largest-magnitude weights, and saves two figures.
"""
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import MultipleLocator
from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import StandardScaler

# Configuration
data_path = "./data/ABIDE2_ASD_7net.csv"
candidate_labels = ["VN", "SMN", "DAN", "VAN", "FPN", "DMN"]
candidate_ados = [
    "SRS_TOTAL_RAW", "SRS_COGNITION_RAW", "SRS_COMMUNICATION_RAW",
    "SRS_MOTIVATION_RAW", "SRS_MANNERISMS_RAW"
]
n_components = 1
n_perm = 1000

# Output directory
current_dir = os.path.dirname(os.path.abspath(__file__))
target_file_path = os.path.join(current_dir, "figure")
os.makedirs(target_file_path, exist_ok=True)

# Load data and retain selected columns present in the CSV
df = pd.read_csv(data_path)
brain_cols = [col for col in candidate_labels if col in df.columns]
behavior_cols = [col for col in candidate_ados if col in df.columns]
if not brain_cols or not behavior_cols:
    raise ValueError("No matching brain or behavioral columns were found in the input CSV.")

# Convert to numeric, replace -9999 with NaN, and remove incomplete cases
for col in brain_cols + behavior_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")
    df[col] = df[col].replace(-9999, np.nan)
before = df.shape[0]
df = df.dropna(subset=brain_cols + behavior_cols).copy()
print(f"删除缺失值样本：{before - df.shape[0]} 个")
print(f"纳入分析样本：{df.shape[0]} 个")
print(f"脑网络指标：{brain_cols}")
print(f"行为指标：{behavior_cols}")
if df.shape[0] < 2:
    raise ValueError("Too few complete cases for CCA.")

# Standardize brain and behavioral variables separately
X = StandardScaler().fit_transform(df[brain_cols].values)
Y = StandardScaler().fit_transform(df[behavior_cols].values)

# Fit CCA and preserve the original scores and weights
cca = CCA(n_components=n_components, max_iter=1000)
cca.fit(X, Y)
x_proj, y_proj = cca.transform(X, Y)
original_x_weights = cca.x_weights_.copy()
original_y_weights = cca.y_weights_.copy()
true_corrs = [
    np.corrcoef(x_proj[:, i], y_proj[:, i])[0, 1]
    for i in range(n_components)
]

# Permutation test: shuffle behavioral rows to break brain-behavior correspondence
perm_corrs = np.zeros((n_perm, n_components))
for p in range(n_perm):
    Y_perm = np.random.permutation(Y)
    cca_perm = CCA(n_components=n_components, max_iter=1000)
    cca_perm.fit(X, Y_perm)
    x_p, y_p = cca_perm.transform(X, Y_perm)
    for i in range(n_components):
        perm_corrs[p, i] = np.corrcoef(x_p[:, i], y_p[:, i])[0, 1]

# Report canonical correlations and permutation p-values
for i in range(n_components):
    r_true = true_corrs[i]
    p_val = (np.sum(perm_corrs[:, i] >= r_true) + 1) / (n_perm + 1)
    print(f"[Component {i + 1}] r = {r_true:.4f} | p = {p_val:.4f} "
          f"{'ok' if p_val < 0.05 else 'no'}")

# Print the six brain-network weights with the largest absolute values
for i in range(n_components):
    weights = original_x_weights[:, i]
    abs_sorted_idx = np.argsort(np.abs(weights))[::-1]
    print(f"\\nComponent {i + 1} — Top 6 brain regions:")
    for j in abs_sorted_idx[:6]:
        print(f"- {brain_cols[j]:35s} : weight = {weights[j]:.4f}")

# Print the five behavioral weights with the largest absolute values