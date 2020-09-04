#!/bin/env python
import math,sys,argparse,pickle

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from packaging import version
from vspaper_settings import paper_palettes, name_map, reverse_map, swarm_markers

# requires pandas>=1.1.0 because of list arg to pandas::DataFrame::pivot index
assert version.parse(pd.__version__) >= version.parse('1.1.0'), 'pandas version >= 1.1.0 required'

def plot_iou():
    '''
    you pass in the groupby where you've done any desired selections (e.g.
    based on label value)
    '''
    pass

parser = argparse.ArgumentParser(description='Compute various kinds of '
        'correlation/intersection between different methods')
parser.add_argument('-f', '--files', nargs='+', help='Summary files to compute '
        'correlation from, layout should be TRUE PREDICTION TARGET TITLE METHOD')
parser.add_argument('-a', '--affinity', action='store_true', help='Indicate '
        'that the values are reporting affinity rather than label, and therefore '
        'only correlations will be computed')

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
    
    # intersection/union for top 1% of actives per target
    # print('Intersection over union for top 1% of actives per target:\n')
    targets = df['Target'].unique()
    mnames = list(df['Method'].unique())
    backup_palette = sns.color_palette("hls", n_colors=len(mnames),
            desat=.5).as_hex()
    iou_data = {'Method1':[], 'Method2':[], 'Target':[], 'IOU':[]}
    palette = {}
    for mn in mnames:
        if mn in name_map:
            method1 = name_map[mn]
        else:
            method1 = mn.replace('_', ' ')
        palette[method1] = paper_palettes[method1] if method1 in \
            paper_palettes else backup_palette[mnames.index(mn)]
        grouped = df.loc[df['True']==1].groupby(['Method', 'Target'])
        topn = grouped.apply(lambda x: x.nlargest(math.ceil(len(x) * .01), 'Prediction'))
        topn.reset_index(inplace=True, drop=True)
        thism_df = topn.loc[topn['Method']==mn]
        # TODO: uuuuuuuuuughhhhhhhh
        for on in mnames:
            if mn == on:
                continue
            if on in name_map:
                method2 = name_map[on]
            else:
                method2 = on.replace('_', ' ')
            thatm_df = topn.loc[topn['Method']==on]
            for target in targets:
                thism = set(thism_df.loc[thism_df['Target']==target]['Title'])
                thatm = set(thatm_df.loc[thatm_df['Target']==target]['Title'])
                iou = len(thism.intersection(thatm)) / len(thism.union(thatm))
                iou_data['Method1'].append(method1)
                iou_data['Method2'].append(method2)
                iou_data['Target'].append(target.replace('_', ' '))
                iou_data['IOU'].append(iou)

    iou_data = pd.DataFrame(iou_data)
    print(iou_data.to_markdown())
    fig,ax = plt.subplots(figsize=(12.8,9.6))
    grouped = iou_data.groupby(['Method1'], as_index=False)
    medians = grouped['IOU'].median()
    medians.sort_values(by='IOU', inplace=True)
    order = medians['Method1'].tolist()
    targets = list(iou_data['Target'].unique()) # might have changed when we replaced underscores
    if len(targets) <= 20:
        leghands = []
        for marker_id,target in enumerate(targets):
            if marker_id > 11:
                mew = 2
                size = 8
            else:
                mew = 0.5
                size = 6
            marker = swarm_markers[marker_id]
            sns.stripplot(x='Method1', y='IOU', hue='Method2',
                    data=iou_data[iou_data['Target']==target],
                    split=True, size=size,
                    jitter = True, 
                    linewidth=mew,
                    alpha=0.7, 
                    palette=palette, marker=marker,
                    ax=ax, order=order)
            leghands.append(mlines.Line2D([], [], color='black',
                fillstyle='none', marker=marker, linestyle='None',
                mew=1,
                markersize=size, label=target))
            sns.boxplot(x='Method1', y='IOU', data=iou_data,
                    color='white', ax=ax, order=order)
            ax.legend(handles=leghands, bbox_to_anchor=(1.3, 1),
                    frameon=True, loc='upper right')
            ax.set_ylabel('Intersection Over Union\nTop 1\% of Actives')
            ax.set_xlabel('')
            fig.savefig('topactives_differentmarkers_iou_boxplot.pdf', bbox_inches='tight')
    else:
        sns.boxplot(x='Method1', y='IOU', data=iou_data,
                color='white', ax=ax, order=order)
        sns.swarmplot(x='Method1', y='IOU', hue='Method2',
                data=iou_data, split=True, edgecolor='black', size=7,
                linewidth=0, palette = palette, ax=ax,
                alpha=0.7, order=order)
        ax.set_ylabel('Intersection Over Union\nTop 1\% of Actives')
        ax.set_xlabel('')
        fig.savefig('topactives_iou_boxplot.pdf', bbox_inches='tight')
    
    # intersection/union for bottom 1% of actives per target
    # print('Intersection over union for bottom 1% of actives per target:\n')

    # intersection/union for top 1% of inactives per target
    # print('Intersection over union for top 1% of inactives per target:\n')

    # intersection/union for bottom 1% of inactives per target
    # print('Intersection over union for bottom 1% of inactives per target:\n')
