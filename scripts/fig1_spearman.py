# -*- coding: utf-8 -*-
"""scripts/fig1_spearman.py

Spearman rank correlations among the three n-type input features
(m_n, PF_n, S_n) across all 9,036 compounds (manuscript Figure 1, Section 3.2).

The computation is deterministic (no random numbers). Outputs:
  results/Spearman_Correlation.csv   rho and two-tailed p-values
  figures/Fig1_Spearman_Heatmap.png  heatmap (lower triangle)

Run from the repository root:  python scripts/fig1_spearman.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr

data_dir, results_dir, figures_dir = 'data', 'results', 'figures'
os.makedirs(results_dir, exist_ok=True)
os.makedirs(figures_dir, exist_ok=True)

df = pd.read_csv(os.path.join(data_dir, 'boltztrap_mp.csv'))
features = ['m_n', 'PF_n', 'S_n']
X = df[features]
n = len(X)

k = len(features)
rho = np.eye(k)
pval = np.zeros((k, k))
for i in range(k):
    for j in range(k):
        if i != j:
            r, p = spearmanr(X[features[i]], X[features[j]])
            rho[i, j], pval[i, j] = r, p

rows = []
for i in range(k):
    for j in range(i + 1, k):
        rows.append({'feature_1': features[i], 'feature_2': features[j],
                     'spearman_rho': round(rho[i, j], 4), 'p_value_two_tailed': pval[i, j]})
        print(f"rho({features[i]}, {features[j]}) = {rho[i, j]:+.3f}  (p = {pval[i, j]:.2e}, n = {n:,})")
pd.DataFrame(rows).to_csv(os.path.join(results_dir, 'Spearman_Correlation.csv'), index=False)

annot = np.empty((k, k), dtype=object)
for i in range(k):
    for j in range(k):
        if i == j:
            annot[i, j] = '1.00'
        else:
            stars = '***' if pval[i, j] < 0.001 else '**' if pval[i, j] < 0.01 else '*' if pval[i, j] < 0.05 else 'ns'
            annot[i, j] = f'{rho[i, j]:.2f}\n{stars}'

labels = [r'$m_n$', r'$PF_n$', r'$S_n$']
fig, ax = plt.subplots(figsize=(3.8, 3.5), dpi=300)
sns.heatmap(rho, ax=ax, mask=np.triu(np.ones_like(rho, dtype=bool), k=1),
            cmap=sns.diverging_palette(220, 10, as_cmap=True), vmin=-1, vmax=1, center=0,
            annot=annot, fmt='', annot_kws={'size': 11, 'weight': 'bold', 'color': 'black'},
            linewidths=0.8, linecolor='white', square=True,
            xticklabels=labels, yticklabels=labels,
            cbar_kws={'shrink': 0.85, 'label': r'Spearman $\rho$', 'ticks': [-1, -0.5, 0, 0.5, 1]})
ax.set_yticklabels(labels, rotation=0)
ax.tick_params(axis='both', length=0)
ax.text(0.5, -0.15, r'*** $p < 0.001$', transform=ax.transAxes, ha='center', va='top', fontsize=9)
plt.tight_layout()
out = os.path.join(figures_dir, 'Fig1_Spearman_Heatmap.png')
plt.savefig(out, dpi=300, bbox_inches='tight')
plt.close()
print('Saved:', out)
