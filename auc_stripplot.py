#!/usr/bin/env python
import matplotlib as mpl
mpl.use('Agg')
import os,sys,math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

# calling this here means the above is just being overridden anyway...
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
paper_palettes['Cross-Docked (Pose)'] = '#E59866'
paper_palettes['Cross-Docked (Affinity)'] = '#BA4A00'
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

# you pass in the auc_bytarget files, we make a stripplot by target
df = pd.DataFrame()
for csv in sys.argv[1:]:
    method = csv.split('_bytarget')[0]
    try:
        method = name_map[method]
    except KeyError:
        pass
    names = ['Target', 'AUC', 'APS']
    tmp = pd.read_csv(csv, delim_whitespace=True, header=None, names=names)
    tmp['Method'] = method
    df = pd.concat([df, tmp], ignore_index=True)

df['Target'] = df['Target'].replace('_', ' ', regex=True)
# now sort targets by increasing median AUC
grouped = df.groupby(['Target'], as_index=False)
medians = grouped['AUC'].median()
medians.sort_values(by='AUC', inplace=True)
sorter = medians['Target'].tolist()
sorter_index = dict(zip(sorter,range(len(sorter))))
df['tmp_rank'] = df['Target'].map(sorter_index)
df.sort_values(['tmp_rank'], ascending=True, inplace=True)
df.drop('tmp_rank', 1, inplace=True)

fig,ax = plt.subplots(figsize=(16, 16))
sns.stripplot(x="AUC", y="Target", hue="Method", data=df,
        palette=paper_palettes, alpha=0.7, size=10, ax=ax)
ax_ylims = ax.get_ylim()
ax.plot([0.5, 0.5], [ax_ylims[0], ax_ylims[1]], linestyle='--', color='gray', lw=3,
                    zorder=1, alpha=0.5)
ax.set_ylim(ax_ylims)
ax.legend(title='Method', frameon=True, loc='lower left', ncol=3)
fig.savefig('AUC_stripplot.pdf', bbox_inches='tight')
