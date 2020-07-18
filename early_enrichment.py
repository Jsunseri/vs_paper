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

SMALL_SIZE=16
MEDIUM_SIZE=20
BIGGER_SIZE=24
SUBFIG_SIZE=24

plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

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
paper_palettes['dense-CNNaffinity-mean'] = '#28B463'
paper_palettes['dense-aff-mean'] = '#28B463'
paper_palettes['crossdock_default2018-CNNscore-mean'] = '#E59866'
paper_palettes['crossdock_default2018-CNNaffinity-mean'] = '#BA4A00'
paper_palettes['crossdocked2018-CNNaffinity-mean'] = '#BA4A00'
paper_palettes['general_default2018-CNNscore-mean'] = '#D7BDE2'
paper_palettes['general_default2018-CNNaffinity-mean'] = '#9B59B6'
paper_palettes['generalset2018-CNNaffinity-mean'] = '#9B59B6'
paper_palettes['rf-score-vs'] = '#D98880'
paper_palettes['rf-score-4'] = '#A93226'
paper_palettes['Dense (Pose)'] = '#82E0AA'
paper_palettes['Dense (Affinity)'] = '#28B463'
paper_palettes['Cross-Docked Set (Pose)'] = '#E59866'
paper_palettes['Cross-Docked Set (Affinity)'] = '#BA4A00'
paper_palettes['General Set (Pose)'] = '#D7BDE2'
paper_palettes['General Set (Affinity)'] = '#9B59B6'
paper_palettes['RFScore-VS'] = '#5DADE2'
paper_palettes['RFScore-4'] = '#2874A6'

name_map = {'dense-CNNscore-mean': 'Dense (Pose)', 'dense-CNNaffinity-mean': 'Dense (Affinity)',
        'crossdocked_default2018-CNNscore-mean': 'Cross-Docked Set (Pose)', 
        'crossdock_default2018-CNNaffinity-mean': 'Cross-Docked Set (Affinity)', 
        'general_default2018-CNNscore-mean': 'General Set (Pose)',
        'general_default2018-CNNaffinity-mean': 'General Set (Affinity)', 
        'rfscore-vs': 'RFScore-VS',
        'rf-score-4': 'RFScore-4',
        'dense-aff-mean': 'Dense (Affinity)',
        'crossdocked2018-CNNaffinity-mean': 'Cross-Docked Set (Affinity)', 
        'generalset2018-CNNaffinity-mean': 'General Set (Affinity)'}

# preds are LABELS PREDICTIONS TARGET METHOD 
# only args are summary files
cols = ['Labels', 'Predictions', 'Target', 'Method']
# dict of baseline enrichment for each target
# TODO: compare all summary files on this as a sanity check; right now i can
# see the numbers of actives/total compounds are off for several targets but it isn't obvious to me
# why, some of the time it's clearly including crystal ligands, might also be an issue with compound titles?
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
    if method in ['Vina', 'Vinardo', 'RFScore-VS', 'RFScore-4']:
        EF1['Method'] = method
    else:
        EF1['Method'] = name_map[method]
    if i > 0:
        allEFs = pd.concat([allEFs, EF1])
    else:
        allEFs = EF1.copy()

allEFs.reset_index(level='Target', inplace=True)
allEFs['Target'] = allEFs['Target'].replace('_', ' ', regex=True)
fig,ax = plt.subplots()
sns.barplot(x="Target", y="EF1\%", data=allEFs, ax=ax, hue="Method",
        palette=paper_palettes)
for tick in ax.get_xticklabels():
    tick.set_rotation(90)
ax.legend(bbox_to_anchor=(2, 1.05), frameon=True, loc='upper right')
fig.savefig("enrichment_barplot.pdf", bbox_inches="tight")
