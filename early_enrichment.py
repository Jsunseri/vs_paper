#!/usr/bin/env python
import matplotlib as mpl
mpl.use('Agg')
import os,sys,math
import collections
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.lines as mlines

# calculate early enrichment, EFsubset = {Ligands_selected/Nsubset} / {Ligandstotal/Ntotal},
# for each target + method, and compare with a baseline of EF1% = 2

SMALL_SIZE=10
MEDIUM_SIZE=12
BIGGER_SIZE=12
SUBFIG_SIZE=12

plt.rc('font', size=BIGGER_SIZE)       
plt.rc('axes', titlesize=MEDIUM_SIZE)  
plt.rc('axes', labelsize=MEDIUM_SIZE)  
plt.rc('xtick', labelsize=SMALL_SIZE)  
plt.rc('ytick', labelsize=SMALL_SIZE)  
plt.rc('legend', fontsize=SMALL_SIZE)  
plt.rc('figure', titlesize=BIGGER_SIZE)

plt.style.use('seaborn-white')
mpl.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
mpl.rc('text', usetex=True)

paper_palettes = {}
paper_palettes['Vina'] = '#000000' #the original CNN paper used ccbb44
paper_palettes['CSAR'] = '#332288'
paper_palettes['DUD-E'] = '#4477aa'
paper_palettes['2:1'] = '#88ccee'
paper_palettes['CNN Affinity Rescore'] = '#6da4c0'
paper_palettes['CNN Affinity Refine'] = '#332288'
paper_palettes['CNN Scoring Rescore'] = '#ffd91c'
paper_palettes['CNN Scoring Refine'] = '#877b25'
paper_palettes['Experiment'] = '#498540'
paper_palettes['CNN'] = '#ffd91c'
paper_palettes['CNNscore'] = '#ffd91c'
paper_palettes['CNNaffinity'] = '#6da4c0'
paper_palettes['Overlap'] = sns.color_palette()[0]
paper_palettes['Overlap L2'] = sns.color_palette()[-1]
paper_palettes['Overlap Mult'] = sns.color_palette()[2]
paper_palettes['Overlap Sum'] = sns.color_palette()[4]
paper_palettes['Vinardo'] = '#BDC3C7'
paper_palettes['dense-CNNscore-mean'] = '#82E0AA'
paper_palettes['Dense\n(Pose)'] = '#82E0AA'
paper_palettes['dense-CNNaffinity-mean'] = '#28B463'
paper_palettes['Dense\n(Affinity)'] = '#28B463'
paper_palettes['dense-aff-mean'] = '#28B463'
paper_palettes['dense_consensus'] = '#cdf2dd'
paper_palettes['Dense\n(Consensus)'] = '#cdf2dd'
paper_palettes['crossdock_default2018-CNNscore-mean'] = '#E59866'
paper_palettes['Cross-Docked\n(Pose)'] = '#E59866'
paper_palettes['crossdock_default2018-CNNaffinity-mean'] = '#BA4A00'
paper_palettes['crossdocked2018-CNNaffinity-mean'] = '#BA4A00'
paper_palettes['Cross-Docked\n(Affinity)'] = '#BA4A00'
paper_palettes['crossdock_default2018_consensus'] = '#f0c4a7'
paper_palettes['Cross-Docked\n(Consensus)'] = '#f0c4a7'
paper_palettes['general_default2018-CNNscore-mean'] = '#b788cb'
paper_palettes['General\n(Pose)'] = '#b788cb'
paper_palettes['general_default2018-CNNaffinity-mean'] = '#9B59B6'
paper_palettes['General\n(Affinity)'] = '#9B59B6'
paper_palettes['generalset2018-CNNaffinity-mean'] = '#9B59B6'
paper_palettes['general_default2018_consensus'] = '#e1d2e9'
paper_palettes['General\n(Consensus)'] = '#e1d2e9'
paper_palettes['rf-score-vs'] = '#D98880'
paper_palettes['rf-score-4'] = '#A93226'
paper_palettes['Dense (Pose)'] = '#82E0AA'
paper_palettes['Dense (Affinity)'] = '#28B463'
paper_palettes['Cross-Docked\n(Pose)'] = '#E59866'
paper_palettes['Cross-Docked\n(Affinity)'] = '#BA4A00'
paper_palettes['General (Pose)'] = '#b788cb'
paper_palettes['General (Affinity)'] = '#9B59B6'
paper_palettes['RFScore-VS'] = '#5DADE2'
paper_palettes['RFScore-4'] = '#2874A6'

name_map = {'dense-CNNscore-mean': 'Dense\n(Pose)', 'dense-CNNaffinity-mean': 'Dense\n(Affinity)',
        'crossdocked_default2018-CNNscore-mean': 'Cross-Docked\n(Pose)', 
        'crossdock_default2018-CNNscore-mean': 'Cross-Docked\n(Pose)', 
        'crossdock_default2018-CNNaffinity-mean': 'Cross-Docked\n(Affinity)', 
        'general_default2018-CNNscore-mean': 'General\n(Pose)',
        'general_default2018-CNNaffinity-mean': 'General\n(Affinity)', 
        'rfscore-vs': 'RFScore-VS',
        'rf-score-4': 'RFScore-4',
        'dense-aff-mean': 'Dense\n(Affinity)',
        'crossdocked2018-CNNaffinity-mean': 'Cross-Docked\n(Affinity)', 
        'generalset2018-CNNaffinity-mean': 'General\n(Affinity)', 
        'dense_consensus': 'Dense\n(Consensus)', 
        'crossdock_default2018_consensus': 'Cross-Docked\n(Consensus)', 
        'general_default2018_consensus': 'General\n(Consensus)'}

def sortedgroupedbar(ax, x,y, groupby, data=None, width=0.7, palette=None, **kwargs):
    order = np.zeros(len(data))
    df = data.copy()
    for xi in np.unique(df[x].values):
        group = data[df[x] == xi]
        a = group[y].values
        b = sorted(np.arange(len(a)),key=lambda x:a[x])
        c = sorted(np.arange(len(a)),key=lambda x:b[x])
        order[data[x] == xi] = c   
    df["order"] = order
    _, name_idx = np.unique(df[x].values, return_index=True)
    u = df[x].values[np.sort(name_idx)].tolist()
    # we want indices per-target that sort them, numpy.unique doesn't preserve order
    df["ind"] = df[x].map(u.index)
    # u, df["ind"] = np.unique(df[x].values, return_inverse=True)
    groupnames = np.unique(df[groupby].values)
    ngroups = len(groupnames)
    step = width/ngroups
    if not palette:
        palette = {}
        backup_palette = sns.color_palette("hls", n_colors=ngroups, desat=.5).as_hex()
        for idx,name in enumerate(groupnames):
            palette[name] = backup_palette[idx]
    for xi,grp in df.groupby(groupby):
        ax.bar(grp["ind"].values-width/2.+grp["order"].values*step+step/2.,
               grp[y].values, width=step, label=xi, color=palette[xi], **kwargs)
    ax.legend(title=groupby, frameon=True, loc='upper left', ncol=3)
    ax.set_xticks(np.arange(len(u)))
    ax.set_xticklabels(u)
    labels = ax.get_yticks().tolist()
    labels[0] = ""
    ax.set_yticklabels(labels)
    num_xs = len(grp["Target"].values)
    ax.set_xlim((-0.25, num_xs+0.25))
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax_xlims = ax.get_xlim()
    ax.plot([ax_xlims[0],ax_xlims[1]],[1, 1], linestyle='--', color='gray', lw=5,
                        zorder=1, alpha=0.5)

# preds are LABELS PREDICTIONS TARGET METHOD 
# only args are summary files
cols = ['Labels', 'Predictions', 'Target', 'Method']
# dict of baseline enrichment for each target
baseline_enrichment = {}
for i,fname in enumerate(sys.argv[1:]):
    df = pd.read_csv(fname, delim_whitespace=True, header=None, names=cols)
    method = df['Method'].unique()[0]
    df.sort_values(['Predictions'], ascending=False, inplace=True)
    # for each target, compute actives / total overall
    # and actives in 1% / total in 1% after sorting
    grouped = df.groupby(['Target'])
    actives = grouped.agg({'Labels': 'sum'})
    ncompounds = grouped.size()
    overall = actives.div(ncompounds, level='Target', axis=0)
    # len(x) is the total number of compounds in the group, we want the top 1%
    # of predictions in each group and then we want to see how many of those
    # are actives
    topn = grouped.apply(lambda x: x.nlargest(int(len(x) * .01), 'Predictions'))
    # we still have a Target column, so drop the redundant index
    topn.reset_index(inplace=True, drop=True)
    topn_grouped = topn.groupby(['Target'])
    enrich_actives = topn_grouped.agg({'Labels': 'sum'})
    enrich_total = topn_grouped.size()
    enrich_frac = enrich_actives.div(enrich_total, level='Target', axis=0)
    EF1 = enrich_frac / overall
    EF1.rename(columns={'Labels': 'EF1\%'}, inplace=True)
    if method in ['Vina', 'Vinardo', 'RFScore-VS', 'RFScore-4'] or method not in name_map:
        EF1['Method'] = method
    else:
        EF1['Method'] = name_map[method]
    if i > 0:
        allEFs = pd.concat([allEFs, EF1])
    else:
        allEFs = EF1.copy()

allEFs.reset_index(level='Target', inplace=True)
allEFs['Target'] = allEFs['Target'].replace('_', ' ', regex=True)

fig,ax = plt.subplots(figsize=(12.8,9.6))
grouped = allEFs.groupby(['Method'], as_index=False)
medians = grouped['EF1\%'].median()
medians.sort_values(by='EF1\%', inplace=True)
order = medians['Method'].tolist()
sns.swarmplot(x='Method', y='EF1\%',
        data=allEFs, split=True, edgecolor='black', size=7,
        linewidth=0, palette = paper_palettes, ax=ax,
        alpha=0.7, order=order)
sns.boxplot(x='Method', y='EF1\%', data=allEFs,
        color='white', ax=ax, order=order)
ax_xlims = ax.get_xlim()
ax.plot([ax_xlims[0],ax_xlims[1]],[1, 1], linestyle='--', color='gray',
                    zorder=1, alpha=0.5)
#sigh
ax.set_ylabel('EF1\%')
ax.set_xlabel('')
fig.savefig('EF1_boxplot.pdf', bbox_inches='tight')

SMALL_SIZE=90
MEDIUM_SIZE=92
BIGGER_SIZE=94
SUBFIG_SIZE=96
plt.rc('font', size=BIGGER_SIZE)        
plt.rc('axes', titlesize=MEDIUM_SIZE)   
plt.rc('axes', labelsize=MEDIUM_SIZE)   
plt.rc('xtick', labelsize=SMALL_SIZE)   
plt.rc('ytick', labelsize=SMALL_SIZE)   
plt.rc('legend', fontsize=SMALL_SIZE)   
plt.rc('figure', titlesize=BIGGER_SIZE) 

fig,ax = plt.subplots(figsize=(500,80))
# sort by target with increasing median EF1%, then in the sorted grouped barplot we'll also sort by method 
# within the target. this is all a pain in the butt, can it be done more efficiently?
grouped = allEFs.groupby(['Target'], as_index=False)
medians = grouped['EF1\%'].median()
medians.sort_values(by='EF1\%', inplace=True)
sorter = medians['Target'].tolist()
sorter_index = dict(zip(sorter,range(len(sorter))))
allEFs['tmp_rank'] = allEFs['Target'].map(sorter_index)
allEFs.sort_values(['tmp_rank'], ascending=True, inplace=True)
allEFs.drop('tmp_rank', 1, inplace=True)

sortedgroupedbar(ax, x="Target", y="EF1\%", groupby="Method", data=allEFs, width=0.7, palette=paper_palettes)
# grouped = allEFs.groupby(['Target'], as_index=False)
# medians = grouped['EF1\%'].median()
# medians.sort_values(by='EF1\%', inplace=True)
# order = medians['Target'].tolist()
# sns.swarmplot(x='Target', y='EF1\%',
#         data=allEFs, split=True, edgecolor='black', size=7,
#         linewidth=0, palette = paper_palettes, ax=ax, hue='Method', 
#         alpha=0.7, order=order)
# sns.boxplot(x='Target', y='EF1\%', data=allEFs,
#         color='white', ax=ax, order=order)
#sigh
# ax.set_ylabel('EF1\%')
# ax.set_xlabel('')
fig.savefig('EF1_targets_boxplot.pdf', bbox_inches='tight')
