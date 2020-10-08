#!/usr/bin/env python
import glob, sys, os, argparse

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

import seaborn as sns
import pandas as pd
from tabulate import tabulate

from vspaper_settings import paper_palettes, backup_palette, name_map, reverse_map, swarm_markers, litpcba_ntemplates, litpcba_order, marker_sizes

# let's do clustered barplots for top1/top3/top5 averaged across targets
# because we had that previously
# also do boxplots per method, with points per target, showing fraction of
# compounds that had a < 2A pose by rank N (with N \in {1,3,5}) and have the
# legend show the number of templates per target
# then also do correlation plot for top1/top3/top5 RMSD fraction vs AUC/NEF

# need it for select bolded text in the legend
# mpl.rc('text', usetex=True)
# mpl.rc('text.latex', preamble=r'\usepackage[utf8x]{inputenc}')
# mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['mathtext.fontset'] = 'custom'
mpl.rcParams['mathtext.rm'] = 'STIXGeneral'
mpl.rcParams['mathtext.sf'] = 'DejaVu Sans'

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
parser.add_argument('-o', '--outprefix', default='', help='Output prefix for figure names')
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
        tmp_dfs[target] = pd.merge(tmp_dfs[target], this_df, on=infocols + ['Vina'], sort=False, how='outer')
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
targets = df['Target'].unique().tolist()
grp = df.groupby(['Target', 'Compound'])
vina_mins = df.groupby(['Target', 'Compound'], as_index=False).RMSD.min()
vina_mins = vina_mins.astype({'Target': 'category'})
vina_num_goodposes = vina_mins.loc[vina_mins['RMSD'] <=
        args.threshold].groupby(['Target']).size().reset_index(name='Best Available\n(Vina)')
vina_num_compounds = vina_mins.groupby(['Target']).size().reset_index(name='Number of Templates')
best_summary = pd.merge(vina_num_goodposes, vina_num_compounds, on=['Target'], sort=False, how='outer')
best_summary['Prediction'] = best_summary['Best Available\n(Vina)'] / best_summary['Number of Templates']
best_summary['Method'] = 'Best Available\n(Vina)'

# for boxplot, build up a final DataFrame of fraction of "good" compounds per
# target for given rank N
# include Method == 'Best Vina' and Method == 'Best Vinardo'
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
                        args.threshold)].groupby('Target').size().reset_index(name='Prediction')
        this_summary['Rank'] = rank
        summary = pd.concat([summary, this_summary], ignore_index=True, sort=False)
    summary['Method'] = method
    try:
        overall_summary = pd.concat([overall_summary, summary], ignore_index=True, sort=False)
    except NameError:
        overall_summary = summary
overall_summary = pd.merge(overall_summary, best_summary[['Target', 'Number of Templates']],
        on=['Target'], sort=False, how='outer')
overall_summary['Prediction'] = overall_summary['Prediction'] / overall_summary['Number of Templates']

# add in a dummy rank column and concat the "best" data for each included rank
for rank in [1,3,5]:
    best_summary['Rank'] = rank
    overall_summary = pd.concat([overall_summary, 
        best_summary[['Target', 'Rank', 'Prediction', 'Method', 'Number of Templates']]], 
        ignore_index=True, sort=False)

# get Vinardo preds too, if we have them
vinardo_df = None
for i,f in enumerate(args.vinardo):
    vinardo_df = pd.concat([vinardo_df, pd.read_csv(f, delim_whitespace=True,
        header=0)], ignore_index=True)
    if i == len(args.vinardo)-1:
        methods.append('Vinardo')
        vinardo_df.loc[:,'Vinardo'] = vinardo_df[['Vinardo']].mul(-1) 
        vinardo_mins = vinardo_df.groupby(['Target', 'Compound'], as_index=False).RMSD.min()
        vinardo_mins = vinardo_mins.astype({'Target': 'category'})
        vinardo_num_goodposes = vinardo_mins.loc[vinardo_mins['RMSD'] <=
            args.threshold].groupby(['Target']).size().reset_index(name='Best Available\n(Vinardo)')
        vinardo_num_compounds = vinardo_mins.groupby(['Target']).size().reset_index(name='Number of Templates')
        best_vinardo_summary = pd.merge(vinardo_num_goodposes,
                vinardo_num_compounds, on=['Target'], sort=False, how='outer')
        best_vinardo_summary['Prediction'] = \
                    best_vinardo_summary['Best Available\n(Vinardo)'] / best_vinardo_summary['Number of Templates']
        best_vinardo_summary['Method'] = 'Best Available\n(Vinardo)'

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
                            args.threshold)].groupby('Target').size().reset_index(name='Prediction')
            this_summary['Rank'] = rank
            vinardo_summary = pd.concat([vinardo_summary, this_summary], ignore_index=True, sort=False)
        vinardo_summary['Method'] = 'Vinardo'
        vinardo_summary = pd.merge(vinardo_summary, best_vinardo_summary[['Target', 'Number of Templates']], 
                on=['Target'], sort=False, how='outer')
        vinardo_summary['Prediction'] = vinardo_summary['Prediction'] / vinardo_summary['Number of Templates']
        for rank in [1,3,5]:
            best_vinardo_summary['Rank'] = rank
            vinardo_summary = pd.concat([vinardo_summary, 
                best_vinardo_summary[['Target', 'Rank', 'Prediction', 'Method', 'Number of Templates']]],
                ignore_index=True, sort=False)
        overall_summary = pd.concat([overall_summary, vinardo_summary], ignore_index=True,
                sort=False)

overall_summary['Target'] = overall_summary['Target'].str.replace('_', ' ', regex=False)
# i use the order to enforce a specific symbol assignment and consistent legends
targets = overall_summary['Target'].unique().tolist()
if len(set(targets).intersection(litpcba_order)) == len(targets):
    targets = [t for t in litpcba_order if t in targets]

methods = overall_summary['Method'].unique().tolist()
palette = {}
for method in methods:
    palette[method] = paper_palettes[method] if method in \
        paper_palettes else backup_palette[methods.index(method) % len(backup_palette)]
for rank in [1,3,5]:
    rank_summary = overall_summary.loc[overall_summary['Rank'] == rank]
    # print(tabulate(rank_summary, headers='keys', tablefmt='psql'))
    if len(targets) <= 20:
        symbol_fig,symbol_ax = plt.subplots(figsize=(18,10))
        grouped = rank_summary.groupby(['Method'], as_index=False)
        medians = grouped['Prediction'].median()
        medians.sort_values(by='Prediction', inplace=True)
        order = medians['Method'].tolist()
        leghands = []
        # fill in info about targets and how we'll display them
        template_info = True
        for target in targets:
            if target not in litpcba_ntemplates:
                template_info = False
                break
        markerdict = {}
        mew = 0.5
        size = 22
        for marker_id,target in enumerate(targets):
            marker = r'$\mathsf{%s}$' % (swarm_markers[marker_id].replace('$',''))
            # marker = swarm_markers[marker_id]
            markerdict[target] = (marker,size)
            if template_info:
                ntemp = str(litpcba_ntemplates[target])
                # if ntemp == '1':
                    # ntemp = r'$\textbf{%s}$' %ntemp
                leghands.append(mlines.Line2D([], [], color='black',
                    fillstyle='none', marker=marker, linestyle='None',
                    mew=mew, 
                    markersize=18, label='%s (%s)' %(target,ntemp)))
            else:
                leghands.append(mlines.Line2D([], [], color='black',
                    fillstyle='none', marker=marker, linestyle='None',
                    mew=mew,
                    markersize=18, label=target))
        _,dummyax = plt.subplots(figsize=(12.8,9.6))
        plt.sca(dummyax)
        g = sns.swarmplot(x='Method', y='Prediction',
                data=rank_summary,
                dodge=True, size=14,
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
            t_order = rank_summary.loc[rank_summary['Method'] == m]['Target'].tolist()
            assert len(points) == len(t_order), ('%d points in PathCollection %d but DataFrame '
            'for method %s has %d points' %(len(points), mnum, m, len(t_order)))
            found_targets = []
            for point in points:
                pred = point[1]
                found = False
                for elem in rank_summary.loc[rank_summary['Method'] == m].itertuples():
                    if round(elem.Prediction,4) == round(pred,4):
                        t = elem.Target
                        if t in found_targets:
                            continue
                        else:
                            found_targets.append(t)
                        found = True
                        break
                if not found:
                    sys.exit('Never found prediction %f for method %s' %(pred, m))
                marker,size = markerdict[t]
                symbol_ax.plot(point[0], pred, ms=size,
                        linestyle='None',
                        mew=mew, alpha=0.7,
                        mec='black',
                        color=palette[m], marker=marker, zorder=3)
        sns.boxplot(x='Method', y='Prediction', data=rank_summary,
                color='white', ax=symbol_ax, order=order, 
                showfliers=False, zorder=2)
        lt_symbol = r'$\mathrm{%s}$' % '\u2264'
        angstrom_symbol = r'$\mathrm{%s}$' % '\u212B'
        symbol_ax.set_ylabel('Fraction of Compounds with Pose %s %s %s RMSD' %(lt_symbol, args.threshold, angstrom_symbol))
        symbol_ax.set_xlabel('')
        symbol_ax.set_title('Best seen by rank %d' %rank)
        symbol_ax.set(ylim=(-0.1,1.1))
        symbol_ax.legend(handles=leghands, bbox_to_anchor=(1.15, 1.015),
                frameon=True, loc='upper right')
        symbol_fig.savefig(args.outprefix+'rank%d_differentmarkers_rmsd_boxplot.pdf' %rank, bbox_inches='tight')
    else:
        fig,ax = plt.subplots(figsize=(12.8,9.6))
        grouped = rank_summary.groupby(['Method'], as_index=False)
        medians = grouped['Prediction'].median()
        medians.sort_values(by='Prediction', inplace=True)
        order = medians['Method'].tolist()
        sns.boxplot(x='Method', y='Prediction', data=rank_summary,
                color='white', ax=ax, order=order, 
                showfliers=False)
        
        sns.swarmplot(x='Method', y='Prediction',
                data=rank_summary, split=True, edgecolor='black', size=7,
                linewidth=0, palette = palette, ax=ax,
                alpha=0.7, order=order)
        ax.set_ylabel('Fraction of Compounds with Pose %s %s %s RMSD' %('$\u2264$', args.threshold, '$\u212B$'))
        ax.set_xlabel('')
        ax.set_title('Best seen by rank %d' %rank)
        ax.set(ylim=(-0.1,1.1))
        symbol_fig.savefig(args.outprefix+'rank%d_rmsd_boxplot.pdf' %rank, bbox_inches='tight')
