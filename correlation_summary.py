#!/bin/env python
import math,sys,argparse,pickle

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from sklearn.metrics import roc_auc_score

from packaging import version
from vspaper_settings import paper_palettes, name_map, reverse_map, swarm_markers, litpcba_order

# requires pandas>=1.1.0 because of list arg to pandas::DataFrame::pivot index
assert version.parse(pd.__version__) >= version.parse('1.1.0'), 'pandas version >= 1.1.0 required'

def plot_intersection(df, palette, prefix='', out='', ylabel='', auc={}):
    '''
    you pass in the groupby where you've done any desired selections (e.g.
    based on label value)
    '''
    intersection_data = {'Method1':[], 'Method2':[], 'Target':[], 'Intersection':[]}
    if auc:
        intersection_data['AUC'] = []
    mnames = list(df['Method'].unique())
    targets = list(df['Target'].unique())
    for mn in mnames:
        thism_df = df.loc[df['Method']==mn]
        if mn in name_map:
            method1 = name_map[mn]
        else:
            method1 = mn.replace('_', ' ')
        for on in mnames:
            if mn == on:
                continue
            if on in name_map:
                method2 = name_map[on]
            else:
                method2 = on.replace('_', ' ')
            thatm_df = df.loc[df['Method']==on]
            for target in targets:
                thism = set(thism_df.loc[thism_df['Target']==target]['Title'])
                thatm = set(thatm_df.loc[thatm_df['Target']==target]['Title'])
                intersection = len(thism.intersection(thatm))
                intersection_data['Method1'].append(method1)
                intersection_data['Method2'].append(method2)
                intersection_data['Target'].append(target.replace('_', ' '))
                intersection_data['Intersection'].append(intersection)
                if auc:
                    intersection_data['AUC'].append(auc[(mn,target)])

    intersection_data = pd.DataFrame(intersection_data)
    grouped = intersection_data.groupby(['Method1'], as_index=False)
    medians = grouped['Intersection'].median()
    medians.sort_values(by='Intersection', inplace=True)
    order = medians['Method1'].tolist()
    targets = list(intersection_data['Target'].unique()) # might have changed when we replaced underscores
    if len(set(targets).intersection(litpcba_order)) == len(targets):
        targets = [t for t in litpcba_order if t in targets]
    # if few targets, make plots with unique symbol for each
    if len(targets) <= 20:
        fig,ax = plt.subplots(figsize=(12.8,9.6))
        leghands = []
        for marker_id,target in enumerate(targets):
            if marker_id > 11:
                mew = 2
                size = 8
            else:
                mew = 0.5
                size = 6
            marker = swarm_markers[marker_id]
            sns.stripplot(x='Method1', y='Intersection', hue='Method2',
                    data=intersection_data[intersection_data['Target']==target],
                    dodge=True, size=size,
                    jitter = True, 
                    linewidth=mew,
                    alpha=0.6, 
                    palette=palette, marker=marker,
                    ax=ax, order=order)
            leghands.append(mlines.Line2D([], [], color='black',
                fillstyle='none', marker=marker, linestyle='None',
                mew=1,
                markersize=size, label=target))
        sns.boxplot(x='Method1', y='Intersection', data=intersection_data,
                color='white', ax=ax, order=order)
        ax.legend(handles=leghands, bbox_to_anchor=(1.2, 0),
                frameon=True, loc='lower right')
        ax.set_ylabel('Per-Method Intersection %s' %ylabel)
        ax.set_xlabel('')
        # ax.set(ylim=(-0.025, 1.025))
        fig.savefig('%sintersection%s_differentmarkers_boxplot.pdf' %(prefix,out), bbox_inches='tight')
    if auc:
        fig,ax = plt.subplots(figsize=(12.8,9.6))
        threshold = 0.6
        relevant = intersection_data.loc[intersection_data.AUC < threshold]
        labels = relevant['Method1'].unique().tolist()
        dummy = {}
        dummy['Method1'] = []
        dummy['Intersection'] = []
        dummy['Method2'] = []
        for mn in order:
            if mn not in labels:
                dummy['Method1'].append(mn)
                dummy['Intersection'].append(-1)
                dummy['Method2'].append('Vina') # doesn't really matter, just
                                                # needs to be a valid method
        if dummy['Method1']:
            relevant = relevant.append(pd.DataFrame(dummy))
        sns.stripplot(x='Method1', y='Intersection',  hue='Method2',
                data=relevant, order=order, 
                palette = palette, 
                jitter = True, 
                dodge=True, 
                size=6, marker='o', 
                linewidth=0.5, ax=ax,
                alpha=0.6)
        relevant = intersection_data.loc[intersection_data.AUC >= threshold]
        labels = relevant['Method1'].unique().tolist()
        dummy = {}
        dummy['Method1'] = []
        dummy['Intersection'] = []
        dummy['Method2'] = []
        for mn in order:
            if mn not in labels:
                dummy['Method1'].append(mn)
                dummy['Intersection'].append(-1)
                dummy['Method2'].append('Vina')
        if dummy['Method1']:
            relevant = relevant.append(pd.DataFrame(dummy))
        sns.stripplot(x='Method1', y='Intersection', hue='Method2', 
                data=relevant, order=order, 
                palette = palette, 
                jitter = True, 
                dodge=True, 
                size=7, marker='*', 
                linewidth=0.5, ax=ax,
                alpha=0.6)
        sns.boxplot(x='Method1', y='Intersection', data=intersection_data,
                color='white', ax=ax, order=order)
        ax.set_ylabel('Per-Method Intersection %s' %ylabel)
        ax.set_xlabel('')
        # ax.set(ylim=(-0.025, 1.025))
        handles,labels = ax.get_legend_handles_labels()
        realnum = int(len(handles)/2)
        box = ax.get_position()
        ax.get_legend().remove()
        fig.legend(handles[:realnum], labels[:realnum], bbox_to_anchor=(1.1,0.1), 
                loc='lower right', frameon=True)
        fig.savefig('%sintersection%s_boxplot.pdf' %(prefix,out), bbox_inches='tight')
    if len(targets) > 20 and not auc:
        fig,ax = plt.subplots(figsize=(12.8,9.6))
        sns.stripplot(x='Method1', y='Intersection', hue='Method2',
                data=intersection_data, dodge=True, size=6, jitter=True, 
                linewidth=0.5, palette = palette, ax=ax,
                alpha=0.6, order=order)
        sns.boxplot(x='Method1', y='Intersection', data=intersection_data,
                color='white', ax=ax, order=order)
        ax.set_ylabel('Per-Method Intersection %s' %ylabel)
        ax.set_xlabel('')
        # ax.set(ylim=(-0.025, 1.025))
        fig.savefig('%sintersection%s_boxplot.pdf' %(prefix,out), bbox_inches='tight')

parser = argparse.ArgumentParser(description='Compute varintersections kinds of '
        'correlation/intersection between different methods')
parser.add_argument('-f', '--files', nargs='+', help='Summary files to compute '
        'correlation from, layout should be TRUE PREDICTION TARGET TITLE METHOD')
parser.add_argument('--auc', action='store_true', help='Use per-target AUC to '
        'set plot symbols')
parser.add_argument('-a', '--affinity', action='store_true', help='Indicate '
        'that the values are reporting affinity rather than label, and therefore '
        'only correlations will be computed')
parser.add_argument('-p', '--prefix', default='', help='Optionally give a '
        'prefix to be added to output files')

args = parser.parse_args()

# you provide the summary files, they need to be 
# LABELS PREDICTIONS TARGET TITLE METHOD
cols = ['True', 'Prediction', 'Target', 'Title', 'Method']
df = pd.DataFrame()
for fname in args.files:
    this_df = pd.read_csv(fname, delim_whitespace=True, header=None, names=cols)
    this_df.drop_duplicates(subset=['True', 'Target', 'Title', 'Method'], inplace=True, ignore_index=True)
    df = pd.concat([df, this_df], ignore_index=True)

try:
    pivoted_df = df.pivot(index=['True', 'Target', 'Title'], columns='Method', values='Prediction')
except Exception as e:
    print(e)
    print('Checking for duplicates...')
    df = df.set_index(['True', 'Target', 'Title'])
    print(df[df.index.duplicated()].to_string())
    sys.exit()

# full correlation
print('Pearson correlation:\n')
print(pivoted_df.corr(method='pearson').to_string())

print('\nSpearman correlation:\n')
print(pivoted_df.corr(method='spearman').to_string())

print('\nKendall Tau correlation:\n')
print(pivoted_df.corr(method='kendall').to_string())

if not args.affinity:
    # actives only
    actives = pivoted_df.loc[1]
    print('\nAll actives, Pearson correlation:\n')
    print(actives.corr(method='pearson').to_string())
    
    print('\nAll actives, Spearman correlation:\n')
    print(actives.corr(method='spearman').to_string())
    
    print('\nAll actives, Kendall Tau correlation:\n')
    print(actives.corr(method='kendall').to_string())
    
    # inactives only
    inactives = pivoted_df.loc[0]
    print('\nAll inactives, Pearson correlation:\n')
    print(inactives.corr(method='pearson').to_string())
    
    print('\nAll inactives, Spearman correlation:\n')
    print(inactives.corr(method='spearman').to_string())
    
    print('\nAll inactives, Kendall Tau correlation:\n')
    print(inactives.corr(method='kendall').to_string())
    
    # TODO: maybe just actives/inactives for targets a given method does well on?
    
    mnames = list(df['Method'].unique())
    backup_palette = sns.color_palette("hls", n_colors=len(mnames),
            desat=.5).as_hex()
    palette = {}
    for mn in mnames:
        if mn in name_map:
            method1 = name_map[mn]
        else:
            method1 = mn.replace('_', ' ')
        palette[method1] = paper_palettes[method1] if method1 in \
            paper_palettes else backup_palette[mnames.index(mn)]

    # compute AUCs if we're using them to select symbols
    grouped = df.groupby(['Method', 'Target'])
    aucs = {}
    if args.auc:
        for name,group in grouped:
            aucs[name] = roc_auc_score(group['True'], group['Prediction'])

    # intersection for top 50 mols per target
    topn = grouped.apply(lambda x: x.nlargest(50, 'Prediction'))
    topn.reset_index(inplace=True, drop=True)
    plot_intersection(topn, palette, args.prefix, '_top50', '\nTop 50 Compounds', aucs)

    # intersection for top 10 actives per target
    grouped = df.loc[df['True']==1].groupby(['Method', 'Target'])
    topn = grouped.apply(lambda x: x.nlargest(10, 'Prediction'))
    topn.reset_index(inplace=True, drop=True)
    plot_intersection(topn, palette, args.prefix, '_topactives', '\nTop 10 Actives', aucs)
    
    # intersection for bottom 10 actives per target
    grouped = df.loc[df['True']==1].groupby(['Method', 'Target'])
    topn = grouped.apply(lambda x: x.nsmallest(10, 'Prediction'))
    topn.reset_index(inplace=True, drop=True)
    plot_intersection(topn, palette, args.prefix, '_worstactives', '\nBottom 10 Actives', aucs)

    # intersection for top 10 inactives per target
    grouped = df.loc[df['True']==0].groupby(['Method', 'Target'])
    topn = grouped.apply(lambda x: x.nlargest(10, 'Prediction'))
    topn.reset_index(inplace=True, drop=True)
    plot_intersection(topn, palette, args.prefix, '_topinactives', '\nTop 10 Inactives', aucs)

    # intersection for bottom 10 inactives per target
    grouped = df.loc[df['True']==0].groupby(['Method', 'Target'])
    topn = grouped.apply(lambda x: x.nsmallest(10, 'Prediction'))
    topn.reset_index(inplace=True, drop=True)
    plot_intersection(topn, palette, args.prefix, '_worstinactives', '\nBottom 10 Inactives', aucs)
