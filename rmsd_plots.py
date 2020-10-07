#!/usr/bin/env python
import glob, sys, os, argparse

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import seaborn as sns
import pandas as pd

from vspaper_settings import paper_palettes, backup_palette, name_map, reverse_map, swarm_markers, litpcba_ntemplates, litpcba_order

# let's do clustered barplots for top1/top3/top5 averaged across targets
# because we had that previously
# also do boxplots per method, with points per target, showing fraction of
# compounds that had a < 2A pose by rank N (with N \in {1,3,5}) and have the
# legend show the number of templates per target
# then also do correlation plot for top1/top3/top5 RMSD fraction vs AUC/NEF

# need it for select bolded text in the legend
mpl.rc('text', usetex=True)

parser = argparse.ArgumentParser(description='Plot summary of pose prediction '
        'performance and correlation between pose prediction and virtual '
        'screening')
parser.add_argument('-s', '--summary', nargs='+', help='Files summarizing '
        'poses, scores, and RMSDs, with a header row on the first line, '
        'including cols [Rank Compound [CNN_NAME]_CNNaffinity [CNN_NAME]_CNNscore '
        'Vina Weight RMSD Target File]')
parser.add_argument('-v', '--vinardo', nargs='*', default=[], help='Files summarizing Vinardo '
        'poses, scores, and RMSDs, with a header row on the first line, '
        'including cols [Rank Compound Vinardo Weight RMSD Target File]')
parser.add_argument('--aucs', nargs='*', default=[], help='AUC by-target '
        'summary files corresponding to the methods in the score summary files')
parser.add_argument('--EFs', nargs='*', default=[], help='Normalized EF1% '
        'summary files corresponding to the methods in the score summary files')
parser.add_argument('-t', '--threshold', default=2.5, type=float,
        help='Threshold defining "good" poses')
args = parser.parse_args()

infocols = ["Rank", "Compound", "Weight", "RMSD", "Target", "File"]

tmp_dfs = {}
for f in args.summary:
    # this is inconvenient, because I hadn't come up with the better summary
    # file method I eventually used for the virtual screening data...
    # build up DataFrames for each target by merging the per-method results,
    # then concat the per-target DataFrames once we have them all
    this_df = pd.read_csv(f, delim_whitespace=True, header=0)
    target = this_df['Target'].unique().tolist()
    assert len(target) == 1, ('Currently requiring each summary file to correspond '
    'to one target and one method')
    target = target[0]
    if target in tmp_dfs:
        tmp_dfs[target] = pd.merge(tmp_dfs[target], this_df, on=infocols + ['Vina'], sort=False)
    else:
        tmp_dfs[target] = this_df

df = pd.concat(tmp_dfs.values(), ignore_index=True)

cols = list(df)
methods = []
if 'Vina' in cols:
    df.loc[:,'Vina'] = df[['Vina']].mul(-1) 
    methods.append('Vina')

# compute ensemble means for CNN predictions
cnn = ["dense", "crossdock_default2018", "general_default2018"]
for d in cnn:
    for stype in ['CNNaffinity', 'CNNscore']:
        snames = []
        for seed in list(range(5)):
            if seed == 0:
                model = d
            else:
                model = "%s_%d" %(d, seed)
            seedname = '%s_%s' %(model, stype)
            if seedname not in cols:
                continue
            snames.append(seedname)
        if not snames:
            continue
        mname = name_map[d + '-' + stype + '-mean']
        df[mname] = df[snames].mean(axis=1)
        methods.append(mname)

# drop the individual ensemble predictions
df = df[infocols + methods]
# df = pd.melt(df, id_vars=infocols, value_vars=methods, var_name='Method', value_name='Prediction')
vina_mins = df.groupby(['Target', 'Compound']).RMSD.min()
vina_num_goodposes = vina_mins.loc[vina_mins <=
        args.threshold].groupby(['Target']).size().reset_index(name='Best Available\n(Vina)')
vina_num_compounds = vina_mins.groupby(['Target']).size().reset_index(name='Number of Templates')
best_summary = pd.merge(vina_num_goodposes, vina_num_compounds, on=['Target'], sort=False)
best_summary['Best Available\nFraction\n(Vina)'] = best_summary['Best Available\n(Vina)'] / best_summary['Number of Templates']

# for boxplot, build up a final DataFrame of fraction of "good" compounds per
# target for given rank N, with cols Target TopN Method
# include Method == 'Best Vina' and Method == 'Best Vinardo'
# groupby method, compute mean of top1/top3/top5/best cols
# TARGET METHOD TOP1_RMSD TOP3_RMSD TOP5_RMSD 
best = 'Cumulative Best RMSD'
for i,method in enumerate(methods):
    df = df.sort_values(method, ascending=False)
    grouped_df = df.groupby(['Target', 'Compound'])
    df[best] = grouped_df.RMSD.cummin()
    df['Rank'] = grouped_df[method].rank(method="dense", ascending=False, axis=1)
    # do this so we get 0s in the counts
    df = df.astype({'Target': 'category'})
    summary = pd.DataFrame()
    for rank in [1,3,5]:
        keepcols = [method,'Rank', best, 'Target']
        this_summary = df[keepcols].loc[(df['Rank'] == rank) & (df[best] <=
                        args.threshold)].groupby('Target').size().reset_index(name=method)
        this_summary['Rank'] = rank
        summary = pd.concat([summary, this_summary], ignore_index=True, sort=False)
    try:
        overall_summary = pd.merge(overall_summary, summary, on=['Target', 'Rank'], sort=False)
    except NameError:
        overall_summary = summary
summary = pd.merge(overall_summary, best_summary, on=['Target'], sort=False)
for method in methods:
    summary[method] = summary[method] / summary['Number of Templates']

# get Vinardo preds too, if we have them
vinardo_df = None
for i,f in enumerate(args.vinardo):
    vinardo_df = pd.concat([vinardo_df, pd.read_csv(f, delim_whitespace=True,
        header=0)], ignore_index=True)
    if i == len(args.vinardo)-1:
        methods.append('Vinardo')
        vinardo_df.loc[:,'Vinardo'] = vinardo_df[['Vinardo']].mul(-1) 
        vinardo_mins = vinardo_df.groupby(['Target', 'Compound']).RMSD.min()
        vinardo_num_goodposes = vinardo_mins.loc[vinardo_mins <=
            args.threshold].groupby(['Target']).size().reset_index(name='Best Available\n(Vinardo)')
        vinardo_num_compounds = vinardo_mins.groupby(['Target']).size().reset_index(name='Number of Templates')
        best_vinardo_summary = pd.merge(vinardo_num_goodposes, vinardo_num_compounds, on=['Target'], sort=False)
        best_vinardo_summary['Best Available\nFraction\n(Vinardo)'] = \
                    best_vinardo_summary['Best Available\n(Vinardo)'] / best_vinardo_summary['Number of Templates']
        vinardo_df = vinardo_df.sort_values('Vinardo', ascending=False)
        grouped_df = vinardo_df.groupby(['Target', 'Compound'])
        vinardo_df[best] = grouped_df.RMSD.cummin()
        vinardo_df['Rank'] = grouped_df['Vinardo'].rank(method="dense", ascending=False, axis=1)
        # do this so we get 0s in the counts
        vinardo_df = vinardo_df.astype({'Target': 'category'})
        vinardo_summary = pd.DataFrame()
        for rank in [1,3,5]:
            keepcols = ['Vinardo', 'Rank', best, 'Target']
            this_summary = vinardo_df[keepcols].loc[(vinardo_df['Rank'] == rank) & (vinardo_df[best] <=
                            args.threshold)].groupby('Target').size().reset_index(name='Vinardo')
            this_summary['Rank'] = rank
            vinardo_summary = pd.concat([vinardo_summary, this_summary], ignore_index=True, sort=False)
        vinardo_summary = pd.merge(vinardo_summary, best_vinardo_summary, on=['Target'], sort=False)
        vinardo_summary['Vinardo'] = vinardo_summary['Vinardo'] / vinardo_summary['Number of Templates']
        summary = pd.merge(summary, vinardo_summary, on=['Target', 'Rank'], sort=False, 
                suffixes=('_Vina', '_Vinardo'))

targets = summary['Target'].unique().tolist()
if len(set(targets).intersection(litpcba_order)) == len(targets):
    targets = [t for t in litpcba_order if t in targets]
if len(targets) <= 20:
    symbol_fig,symbol_ax = plt.subplots(figsize=(12.8,9.6))
    grouped = boxplot_df.groupby(['Method'], as_index=False)
    medians = grouped['AUC'].median()
    medians.sort_values(by='AUC', inplace=True)
    order = medians['Method'].tolist()
    leghands = []
    # fill in info about targets and how we'll display them
    success_info = True
    for target in targets:
        if target not in litpcba_successes:
            success_info = False
            break
    markerdict = {}
    mew = 0.5
    for marker_id,target in enumerate(targets):
        marker = swarm_markers[marker_id]
        if marker in marker_sizes:
            size = marker_sizes[marker]
        else:
            size = 14
        markerdict[target] = (marker,size)
        if success_info:
            leghands.append(mlines.Line2D([], [], color='black',
                fillstyle='none', marker=marker, linestyle='None',
                mew=0.5, #linewidth=1, 
                markersize=size, label='%s (%s)' %(target,' '.join(litpcba_successes[target]))))
        else:
            leghands.append(mlines.Line2D([], [], color='black',
                fillstyle='none', marker=marker, linestyle='None',
                mew=0.5,
                markersize=size, label=target))
    _,dummyax = plt.subplots(figsize=(12.8,9.6))
    plt.sca(dummyax)
    g = sns.swarmplot(x='Method', y='AUC',
            data=boxplot_df,
            dodge=True, size=27,
            linewidth=mew,
            alpha=0.7, 
            palette=palette, 
            edgecolors='none',
            order=order)
    artists = g.get_children()
    offsets = []
    for a in artists:
        if type(a) is mpl.collections.PathCollection:
            offsets.append(a.get_offsets().tolist())
    if not offsets:
        sys.exit('Cannot obtain offsets for scatterplot with unique per-target symbols')
    assert len(order) == len(offsets), ('List of methods is '
    'length %d but offsets is length %d' %(len(order),len(offsets)))
    plt.sca(symbol_ax)
    for mnum,points in enumerate(offsets):
        # each PathCollection will correspond to a method,
        # in the order specified by the "order" variable;
        # i thought 
        # the order within the collection corresponded to
        # the order the targets appear but that seems false so
        # i'm arduously looking it up instead
        m = order[mnum]
        t_order = boxplot_df.loc[boxplot_df['Method'] == m]['Target'].tolist()
        assert len(points) == len(t_order), ('%d points in PathCollection %d but boxplot_dat '
        'for method %s has %d points' %(len(points), mnum, m, len(t_order)))
        for point in points:
            auc = point[1]
            found = False
            for elem in boxplot_dat:
                if (elem['Method'] == m) and (round(elem['AUC'],4) == round(auc,4)):
                    t = elem['Target']
                    found = True
                    break
            if not found:
                sys.exit('Never found AUC %f for method %s' %(auc, m))
            marker,size = markerdict[t]
            symbol_ax.plot(point[0], auc, ms=size,
                    linestyle='None',
                    mew=mew, alpha=0.7,
                    mec='black',
                    color=palette[m], marker=marker, zorder=3)
    sns.boxplot(x='Method', y='AUC', data=boxplot_df,
            color='white', ax=symbol_ax, order=order, 
            showfliers=False, zorder=2)
    symbol_ax.legend(handles=leghands, bbox_to_anchor=(1.27, 1.015),
            frameon=True, loc='upper right')
    # symbol_ax.legend(handles=leghands, loc='lower right', ncol=2, 
            # frameon=True)
    symbol_ax.set_ylabel('AUC')
    symbol_ax.set_xlabel('')
    symbol_ax.set(ylim=(0,1.1))
    symbol_xlims = symbol_ax.get_xlim()
    symbol_ax.plot([symbol_xlims[0],symbol_xlims[1]],[0.5, 0.5], linestyle='--', color='gray',
            zorder=1, alpha=0.5)
    if not args.color_scheme == 'overlap' and not args.color_scheme == 'vspaper':
        for tick in symbol_ax.get_xticklabels():
            tick.set_rotation(90)
    symbol_fig.savefig(args.outprefix+'_differentmarkers_auc_boxplot.pdf', bbox_inches='tight')
else:
    grouped = boxplot_df.groupby(['Method'], as_index=False)
    medians = grouped['AUC'].median()
    medians.sort_values(by='AUC', inplace=True)
    order = medians['Method'].tolist()
    sns.boxplot(x='Method', y='AUC', data=boxplot_df,
            color='white', ax=auc_ax, order=order, 
            showfliers=False)
    
    sns.swarmplot(x='Method', y='AUC',
            data=boxplot_df, split=True, edgecolor='black', size=7,
            linewidth=0, palette = palette, ax=auc_ax,
            alpha=0.7, order=order)
