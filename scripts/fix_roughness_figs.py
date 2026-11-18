"""Regenerate Fig10 (roughness KDE) and Fig16 (HCP vs roughness) with the true roughness scale.
The original notebooks relabelled x ticks (10..70 -> 20..200), which the manuscript text then quoted."""
import sys
import pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt, seaborn as sns, matplotlib as mpl

data, out = sys.argv[1], sys.argv[2]
df = pd.read_csv(data).dropna()
mpl.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
                     'pdf.fonttype': 42, 'axes.linewidth': 0.8, 'xtick.direction': 'out', 'ytick.direction': 'out'})

# Fig10: KDE of surface roughness, same styling as kde_plotting.ipynb, true scale
fig, ax = plt.subplots(figsize=(7, 5))
sns.kdeplot(data=df, x='surface_roughness', fill=True, color='#2d6a2d', facecolor='#8fbc8f', alpha=0.6, linewidth=1.5, ax=ax)
ax.set_xlabel('Surface roughness index (std. dev. of gradient magnitude)', fontsize=10, labelpad=6)
ax.set_ylabel('Density', fontsize=10, labelpad=6)
ax.tick_params(axis='both', labelsize=9, length=4, width=0.8)
fig.patch.set_facecolor('white'); ax.set_facecolor('white')
plt.tight_layout(); fig.savefig(f'{out}/Fig10.pdf', format='pdf', bbox_inches='tight'); plt.close(fig)

# Fig16: HCP vs roughness scatter, same styling as bivariate_analysis.ipynb, true scale
fig, ax = plt.subplots(figsize=(7, 5))
sns.scatterplot(data=df, x='surface_roughness', y='half-cell_potential_v', s=22, color='#d95f02',
                edgecolor='black', linewidth=0.35, alpha=0.70, ax=ax)
ax.set_xlabel('Surface roughness index (std. dev. of gradient magnitude)', fontsize=11, labelpad=8)
ax.set_ylabel('Half-cell potential (V)', fontsize=11, labelpad=8)
ax.tick_params(axis='both', which='major', labelsize=10, length=5, width=0.8)
for s in ax.spines.values(): s.set_visible(True); s.set_linewidth(0.8)
ax.grid(True, linestyle='--', linewidth=0.4, alpha=0.25)
fig.patch.set_facecolor('white'); ax.set_facecolor('white')
plt.tight_layout(); fig.savefig(f'{out}/Fig16.pdf', format='pdf', bbox_inches='tight'); plt.close(fig)
print(df.surface_roughness.describe().round(2).to_dict())
