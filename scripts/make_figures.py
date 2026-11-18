"""Manuscript figures for revision 3: model stability (R1.4, R2.2) and feature-set comparison (R2.1, R3.4)."""
import sys
import pandas as pd, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

out = sys.argv[1]
r = pd.read_csv(f'{out}/repeated_splits_raw.csv')
BLUE, GREY, INK, INK2, GRID = '#2a78d6', '#9a9893', '#0b0b0b', '#52514e', '#e4e3df'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.edgecolor': GRID,
                     'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK, 'axes.spines.top': False,
                     'axes.spines.right': False})

# ---- Figure A: mean +/- SD of R2 and RMSE, all four inputs, 30 splits
a = r[r.feature_set == 'All four inputs'].groupby('model').agg(R2m=('R2', 'mean'), R2s=('R2', 'std'),
                                                             RMm=('RMSE', 'mean'), RMs=('RMSE', 'std'))
a = a.sort_values('R2m')
seed42 = pd.read_csv(f'{out}/table6_seed42.csv').set_index('model')
fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
yy = np.arange(len(a))
for ax, (m, s, lab) in zip(axes, [('R2m', 'R2s', 'R²'), ('RMm', 'RMs', 'RMSE (V)')]):
    for i, (name, row) in enumerate(a.iterrows()):
        c = BLUE if name == 'Random Forest' else GREY
        ax.errorbar(row[m], i, xerr=row[s], fmt='o', ms=7, color=c, ecolor=c, elinewidth=2, capsize=0)
        ax.text(row[m], i + 0.28, f'{row[m]:.3f} ± {row[s]:.3f}', ha='center', va='bottom', fontsize=8.5, color=INK2)
        ref = seed42.loc[name, 'R2' if lab == 'R²' else 'RMSE']
        ax.plot(ref, i, marker='|', ms=12, mew=1.5, color=INK, zorder=3)
    ax.set_xlabel(f'{lab}, mean ± 1 SD over 30 splits')
    ax.grid(axis='x', color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.tick_params(axis='y', length=0)
axes[0].set_yticks(yy, a.index); axes[0].set_ylim(-0.6, len(a) - 0.2)
fig.text(0.01, 0.02, '| = single split used in original Table 6 (seed 42)', fontsize=8.5, color=INK2)
fig.tight_layout(rect=(0, 0.06, 1, 1))
for ext in ('pdf', 'png'):
    fig.savefig(f'{out}/Fig_model_stability.{ext}', dpi=300)

# ---- Figure B: feature-set comparison, RF (accent) and KNN (grey)
order = ['Resistivity only', 'Resistivity + Roughness', 'Resistivity + Temp + Humidity', 'All four inputs']
labels = ['Resistivity only', 'Resistivity + roughness', 'Resistivity + temp. + humidity', 'All four inputs']
b = r.groupby(['feature_set', 'model']).R2.agg(['mean', 'std']).reset_index()
fig, ax = plt.subplots(figsize=(8, 3.6))
for model, c, dy in [('Random Forest', BLUE, 0.12), ('KNN', GREY, -0.12)]:
    d = b[b.model == model].set_index('feature_set').loc[order]
    ys = np.arange(len(order))[::-1] + dy
    ax.errorbar(d['mean'], ys, xerr=d['std'], fmt='o', ms=7, color=c, elinewidth=2, label=model)
    for yv, mv in zip(ys, d['mean']):
        ax.text(mv + 0.012, yv + 0.06, f'{mv:.2f}', fontsize=8.5, color=INK2, va='bottom')
ax.set_yticks(np.arange(len(order))[::-1], labels); ax.tick_params(axis='y', length=0)
ax.set_xlabel('Test R², mean ± 1 SD over 30 random splits'); ax.set_xlim(0.25, 1.0)
ax.grid(axis='x', color=GRID, lw=0.8); ax.set_axisbelow(True)
ax.legend(frameon=False, loc='upper right')
fig.tight_layout()
for ext in ('pdf', 'png'):
    fig.savefig(f'{out}/Fig_feature_sets.{ext}', dpi=300)
print('ok')
