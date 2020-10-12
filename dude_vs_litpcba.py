#!/usr/bin/env python
import math,glob 

from sklearn.metrics import roc_auc_score,roc_curve
from scipy.stats import spearmanr

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import pandas as pd
from vspaper_settings import (paper_palettes, name_map)

mpl.rcParams.update({'mathtext.fontset': 'cm'})
topdir = '/net/pulsar/home/koes/jss97/datasets/lit-pcba'
targets = ['ADRB2', 'MTORC1']
cols = ['Label', 'Prediction', 'Target', 'Title', 'Method']
R = 0.01
props = dict(boxstyle='round', facecolor='white', alpha=0.8)

# let's do ROC curves for each target with AUC and EF1% in the legend
# also hexbins with spearmanr comparing scores between the two docking runs
paper_methods = ['Dense\n(Pose)', 'Dense\n(Affinity)', 
		 'Cross-Docked\n(Pose)', 'Cross-Docked\n(Affinity)',
 		 'General\n(Pose)', 'General\n(Affinity)', 'Vina']

# put DUD-E in first column, LIT-PCBA in the second
roc_fig,roc_ax = plt.subplots(figsize=(16,16))
roc_dict = {}

for i,target in enumerate(targets):
    hex_fig,hex_ax= plt.subplots(figsize=(16,32))
    hex_dict = {}
    correlation_data = {}
    for f in glob.glob('%s/%s/redock_dude/*_summary' %(topdir,target)):
        if 'scorerank' in f:
            continue
        if 'litpcba' in f:
            roc_num = i * 2 + 1
            dataset = 'LIT-PCBA'
        else:
            roc_num = i * 2
            dataset = 'DUD-E'

        thesecols = [n.replace('Prediction', dataset) for n in cols]
        tmpdf = pd.read_csv(f, names=thesecols, delim_whitespace=True)
        method = tmpdf['Method'].unique().tolist()
        assert len(method) == 1, 'More than one method in input file %s' %f
        method = name_map[method[0]]
        try:
            correlation_data[method] = pd.merge(correlation_data[method],
                    tmpdf, on=['Label', 'Target', 'Title', 'Method'],
                    sort=False)
        except KeyError:
            correlation_data[method] = tmpdf

        auc = roc_auc_score(tmpdf['Label'], tmpdf[dataset])
        fpr,tpr,_ = roc_curve(tmpdf['Label'], tmpdf[dataset])
        Nactives = tmpdf['Label'].sum()
        topn = tmpdf.groupby(['Method']).apply(lambda x: x.nlargest(math.ceil(len(x) * R), dataset))
        topn.reset_index(inplace=True, drop=True)
        nactives = topn['Label'].sum()
        ef = nactives / (Nactives * R)
        label = '%s: AUC=%0.3f, EF1%%=%0.3f' %(method.replace('\n', ' '), auc, ef)
        if roc_num not in roc_dict:
            ax = plt.subplot2grid((2,2), (roc_num // 2, roc_num % 2), fig=roc_fig)
            ax.set_aspect('equal')
            roc_dict[roc_num] = ax
        else:
            ax = roc_dict[roc_num]
        ax.plot(fpr, tpr, color=paper_palettes[method], label=label, lw=5,
                zorder=2, alpha=0.7)
    for method,df in correlation_data.items():
        hex_num = paper_methods.index(method)
        if hex_num not in hex_dict:
            ax = plt.subplot2grid((4,2), (hex_num // 2, hex_num % 2), fig=hex_fig)
            ax.set_aspect('equal')#, 'datalim')
        else:
            ax = hex_dict[hex_num]
        ax.hexbin(df['DUD-E'], df['LIT-PCBA'], color=paper_palettes[method],
                mincnt=1, gridsize=50)
        xlims = ax.get_xlim()
        ylims = ax.get_ylim()
        minval = min(xlims[0], ylims[0])
        maxval = max(xlims[1], ylims[1])
        ax.set_xlim(minval, maxval)
        ax.set_ylim(minval, maxval)
        r,_ = spearmanr(df['DUD-E'], df['LIT-PCBA'])
        ax.annotate(r'$\rho = {0:.2}$'.format(r), xy=(.7, .1),
                 xycoords=ax.transAxes, family='serif', bbox=props)
        ax.set_title(method.replace('\n', ' '))
        if hex_num == 0:
            ax.set_ylabel('LIT-PCBA prediction')
        if hex_num // 6:
            ax.set_xlabel('DUD-E prediction')
        hex_fig.savefig('dude_vs_litpcba_%s_correlation.png' %target, dpi=300, bbox_inches='tight')

# add in random performance line and legends for AUC plots
for i in range(4):
    ax = roc_dict[i]
    ax.plot([0, 1], [0, 1], color='gray', lw=5, linestyle='--', zorder=1)
    ax.legend(bbox_to_anchor=(1.05, 1.47))
    if i == 2:
        ax.set_xlabel('DUD-E')
    elif i == 3:
        ax.set_xlabel('LIT-PCBA')
    roc_fig.subplots_adjust(wspace=-0.05, hspace=0.55)
roc_fig.savefig('dude_vs_litpcba_roc.png', dpi=300, bbox_inches='tight')
