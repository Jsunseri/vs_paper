#! /usr/bin/env python
import sys,os
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import pandas as pd
import seaborn as sns
from vspaper_settings import name_map, reverse_map, swarm_markers, marker_sizes

cols = ['Label', 'Prediction', 'Target', 'Title', 'Method']
dataset_map = {'lit-pcba': 'LIT-PCBA', 'dude': 'DUD-E'}
colors = ['#003f5c', '#ffa600']
palette = {}
palette['Vina'] = colors[0]

# i've assumed only two methods for now, Vina and one method to compare with
fnames = sys.argv[1:]
summarydat = {}
summarydat['Target'] = []
summarydat['Method'] = []
summarydat['AUC'] = []
summarydat['Dataset'] = []
for fname in fnames:
    df = pd.read_csv(fname, delim_whitespace=True, header=None, names=cols)
    method = df['Method'].unique()[0]
    if method != 'Vina':
        method = 'CNN'
        palette[method] = colors[1]
    targets = df['Target'].unique().tolist()
    dataset = dataset_map[os.path.basename(os.path.dirname(fname))]
    for target in targets:
        thisdf = df.loc[df['Target'] == target]
        auc = roc_auc_score(thisdf['Label'].to_numpy(), thisdf['Prediction'].to_numpy())
        summarydat['Target'].append(target)
        summarydat['Method'].append(method)
        summarydat['AUC'].append(auc)
        summarydat['Dataset'].append(dataset)

df = pd.DataFrame(summarydat)
fig,ax = plt.subplots(figsize=(4,10))
sns.boxplot(x='Dataset', y='AUC', hue='Method', dodge=True, 
        data=df, color='white', ax=ax, showfliers=False)
ax.get_legend().remove()
for i,box in enumerate(ax.artists):
    box.set_facecolor('white')
sns.swarmplot(x='Dataset', y='AUC',
        data=df, hue='Method', dodge=True, 
        edgecolor='black', size=10, alpha=0.7, 
        linewidth=0.5, 
        palette=palette, 
        ax=ax)
handles,labels = ax.get_legend_handles_labels()
ax.legend(handles=handles[2:], labels=labels[2:], loc='upper right',
        title='Method', borderpad=0.15, handletextpad=0.25)
xlbl = ax.xaxis.get_label()
xlbl.set_bbox(dict(facecolor='none', boxstyle='round', edgecolor='gray', pad=0.15))
xlim = ax.get_xlim()
ylim = ax.get_ylim()
ax.set_ylim((0,ylim[1]))
ax.plot(xlim, [0.5,0.5], linestyle='--', color='gray', zorder=1, alpha=0.5)

fig.savefig('virtualscreen_vina_vs_cnn.pdf', bbox_inches='tight')
