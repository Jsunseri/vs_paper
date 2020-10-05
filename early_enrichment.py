#!/usr/bin/env python
import os,sys,math
import collections
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib as mpl
mpl.use('Agg')

import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

from vspaper_settings import paper_palettes, backup_palette, name_map, reverse_map, swarm_markers, litpcba_successes, litpcba_order

# calculate early enrichment, EFsubset = {actives_selected/Nsubset} / {activestotal/Ntotal},
# or equivalently {# actives in top R} / {# actives in entire library x R}
# for each target + method, and compare with a baseline of EFR = 2
# 
# optionally calculate normalized EF := {# actives in subset} / min{# total actives, # total compounds x R}

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
    ax.plot([ax_xlims[0],ax_xlims[1]],[2, 2], linestyle='--', color='gray', lw=5,
                        zorder=1, alpha=0.5)

core_methods = ['Dense\n(Affinity)', 'Cross-Docked\n(Affinity)', 'General\n(Affinity)', 
                'RFScore-VS', 'RFScore-4', 'Vina', 'Vinardo']
if __name__ == '__main__':
    parser = ArgumentParser(description='Compute early enrichment and/or '
            'normalized enrichment from prediction summary files')
    parser.add_argument('-b', '--barplot', action='store_true', help='Generate '
            'sorted barplot showing the performance of each method vs the '
            'targets')
    parser.add_argument('-n', '--normalized', action='store_true', help='Plot '
            'normalized enrichment as well as standard enrichment')
    parser.add_argument('-a', '--fix_axis', action='store_true', help='Only '
            'used with normalized EF; if passed, fix axis boundaries to [0,1] '
            'rather than shrinking to fit actual distribution (often much '
            'smaller). If you do this, unique points per target will not be used '
            'since it will probably be difficult to resolve them.')
    parser.add_argument('-r', '--ratio', nargs='+', default=[0.01],
            help='Specify one or more ratios to define the size of the subset '
            'for which enrichment will be computed')
    parser.add_argument('-s', '--summaryfiles', nargs='+', help='Provide '
            'one or more prediction summary files that must be structured like '
            '[LABEL PREDICTION TARGET TITLE METHOD]')
    parser.add_argument('-o', '--outprefix', type=str, default='', 
            help='Prefix for output filenames')
    args = parser.parse_args()

    # preds are LABELS PREDICTIONS TARGET TITLE METHOD 
    cols = ['Label', 'Prediction', 'Target', 'Title', 'Method']
    altcols = ['Label', 'Prediction', 'Target', 'Method']
    for R in args.ratio:
        pctg = int(R * 100)
        EFname = 'EF{}%'.format(pctg) # if you set usetex=True you probably have to escape this percent symbol
        for i,fname in enumerate(args.summaryfiles):
            # some of the outputs don't have titles, so check real quick
            with open(fname, 'r') as tmpf:
                for line in tmpf:
                    contents = line.split()
                    if len(contents) == len(cols):
                        usecols = cols
                    elif len(contents) == len(altcols):
                        usecols = altcols
                    else:
                        sys.exit('Unknown columns in input file %s' %fname)
            df = pd.read_csv(fname, delim_whitespace=True, header=None, names=usecols)
            methods = df['Method'].unique()
            nmethods = methods.shape[0]
            assert nmethods == 1, '%s contains data from %d methods' %(fname, nmethods)
            method = methods[0]
            # for each target, compute # actives in subset
            grouped = df.groupby(['Target'], as_index=False)
            actives = grouped.agg({'Label': 'sum'})
            actives.rename(columns={'Label': 'NA'}, inplace=True)
            ncompounds = grouped.size().to_frame('sizeR').reset_index()
            ncompounds['sizeR'] = ncompounds['sizeR'] * R
            ncompounds['sizeR'] = ncompounds['sizeR'].map(math.ceil)
            actives = actives.merge(ncompounds, on='Target')
            actives['min'] = actives.min(numeric_only=True, axis=1)
            # len(x) is the total number of compounds in the group, we want the top 1%
            # of predictions in each group and then we want to see how many of those
            # are actives (which is equal to the sum of the labels)
            topn = grouped.apply(lambda x: x.nlargest(math.ceil(len(x) * R), 'Prediction'))
            # we still have a Target column, so drop the redundant index
            topn.reset_index(inplace=True, drop=True)
            topn_grouped = topn.groupby(['Target'], as_index=False)
            enrich_actives = topn_grouped.agg({'Label': 'sum'})
            enrich_actives.rename(columns={'Label': 'na'}, inplace=True)

            EFR = actives.merge(enrich_actives, on='Target')
            EFR[EFname] = EFR['na'] / (EFR['NA'] * R)
            normalized_name = ''
            if args.normalized:
                normalized_name = 'Normalized EF{}\%'.format(pctg)
                EFR[normalized_name] = EFR['na'] / EFR['min']
            if method in ['Vina', 'Vinardo', 'RFScore-VS', 'RFScore-4']:
                EFR['Method'] = method
            elif method not in name_map:
                EFR['Method'] = method.replace('_', '\n')
            else:
                EFR['Method'] = name_map[method]
            if i > 0:
                allEFs = pd.concat([allEFs, EFR])
            else:
                allEFs = EFR.copy()
        
        # let's dump out a "by target" summary file
        methods = allEFs['Method'].unique()
        for method in methods:
            if args.normalized:
                thismethod = allEFs.loc[allEFs['Method'] == method][['Target', EFname, normalized_name]]
            else:
                thismethod = allEFs.loc[allEFs['Method'] == method][['Target', EFname]]
            if method in ['Vina', 'Vinardo', 'RFScore-VS', 'RFScore-4'] or method not in reverse_map:
                out_method = method
            else:
                out_method = reverse_map[method]
            thismethod.to_csv('%s_%s_bytarget'%(out_method,EFname.replace('\\','')), sep='\t',
                    encoding='utf-8', index=False, header=False)
        
        palette = {}
        lmethods = list(methods)
        for method in lmethods:
            palette[method] = paper_palettes[method] if method in \
                paper_palettes else backup_palette[lmethods.index(method)]

        allEFs['Target'] = allEFs['Target'].replace('_', ' ', regex=True)
        targets = allEFs['Target'].unique()
        ntargets = len(targets)
        success_info = True
        for target in targets:
            if target not in litpcba_successes:
                success_info = False
                break
        if success_info:
            targets = [t for t in litpcba_order if t in targets]
       
        plotcol = [EFname]
        if args.normalized:
            plotcol.append(normalized_name)
        for col in plotcol:
            if len(lmethods) > 5:
                size = (18,10)
            else:
                size = (12, 10)
            fig,ax = plt.subplots(figsize=size)
            grouped = allEFs.groupby(['Method'], as_index=False)
            medians = grouped[col].median()
            medians.sort_values(by=col, inplace=True)
            order = medians['Method'].tolist()
            if (col == normalized_name and args.normalized and not args.fix_axis and ntargets <= 20) or \
                (col == EFname and ntargets <= 20):
                leghands = []
                for marker_id,target in enumerate(targets):
                    mew = 0.5
                    size = 27
                    marker = swarm_markers[marker_id]
                    sns.stripplot(x='Method', y=col,
                            data=allEFs[allEFs['Target']==target],
                            split=True, size=size,
                            jitter = 0.25, 
                            linewidth=mew,
                            alpha=0.7, 
                            palette=palette, marker=marker,
                            ax=ax, order=order)
                    if success_info:
                        leghands.append(mlines.Line2D([], [], color='black',
                            fillstyle='none', marker=marker, linestyle='None',
                            mew=mew,
                            markersize=size, label='%s (%s)' %(target,' '.join(litpcba_successes[target]))))
                    else:
                        leghands.append(mlines.Line2D([], [], color='black',
                            fillstyle='none', marker=marker, linestyle='None',
                            mew=mew,
                            markersize=size, label=target))
                sns.boxplot(x='Method', y=col, data=allEFs,
                        color='white', ax=ax, order=order,
                        showfliers=False)
                ax.legend(handles=leghands, bbox_to_anchor=(1.22, 1.025),
                        frameon=True, loc='upper right')
                if col == EFname:
                    ax_xlims = ax.get_xlim()
                    ax.plot([ax_xlims[0],ax_xlims[1]],[2, 2], linestyle='--', color='gray',
                                        zorder=1, alpha=0.5)
            else:
                sns.swarmplot(x='Method', y=col,
                        data=allEFs, split=True, edgecolor='black', size=7,
                        linewidth=0, palette = palette, ax=ax,
                        alpha=0.7, order=order)
                sns.boxplot(x='Method', y=col, data=allEFs,
                        color='white', ax=ax, order=order, 
                        showfliers=False)
                if col == EFname:
                    ax_xlims = ax.get_xlim()
                    ax.plot([ax_xlims[0],ax_xlims[1]],[2, 2], linestyle='--', color='gray',
                                        zorder=1, alpha=0.5)
                if args.normalized and args.fix_axis and col == normalized_name:
                    lims = ax.get_ylim()
                    ax.set_ylim(lims[0], 1 - lims[0])
            #sigh
            ax.set_ylabel(col)
            ax.set_xlabel('')
            fig.savefig('%s%s_boxplot.pdf' %(args.outprefix, col.replace(' ', '_').replace('\\','')), bbox_inches='tight')
            
            if ntargets > 25:
                size = (16, 30)
            else:
                size = (16, 16)
            fig,ax = plt.subplots(figsize=size)
            # sort by target with increasing median EFR%, then in the sorted grouped barplot we'll also sort by method 
            # within the target. this is all a pain in the butt, can it be done more efficiently?
            grouped = allEFs.groupby(['Target'], as_index=False)
            medians = grouped[col].median()
            medians.sort_values(by=col, inplace=True)
            sorter = medians['Target'].tolist()
            sorter_index = dict(zip(sorter,range(len(sorter))))
            allEFs['tmp_rank'] = allEFs['Target'].map(sorter_index)
            allEFs.sort_values(['tmp_rank'], ascending=True, inplace=True)
            allEFs.drop('tmp_rank', 1, inplace=True)
            
            if success_info:
                allEFs['Target-withinfo'] = allEFs['Target'].apply(lambda x: '%s\n(%s)'
                        %(x, ' '.join(litpcba_successes[x])))
                sns.stripplot(x=col, y="Target-withinfo", hue="Method", data=allEFs,
                        palette=palette, alpha=0.7, size=10, ax=ax)
                ax.set_ylabel('Target')
            else:
                sns.stripplot(x=col, y="Target", hue="Method", data=allEFs,
                        palette=palette, alpha=0.7, size=10, ax=ax)
            if col == EFname:
                ax_ylims = ax.get_ylim()
                ax.plot([2, 2], [ax_ylims[0], ax_ylims[1]], linestyle='--', color='gray', lw=3,
                                    zorder=1, alpha=0.5)
                ax.set_ylim(ax_ylims)
            if args.normalized and args.fix_axis and col == normalized_name:
                lims = ax.get_xlim()
                ax.set_xlim(lims[0], 1 - lims[0])
                ax.legend(title='Method', frameon=True, ncol=3, loc='upper_right')
            else:
                anchor = (1.05, 1)
                ax.legend(title='Method', frameon=True, ncol=3, bbox_to_anchor=anchor)
            handles, labels = ax.get_legend_handles_labels()
            if set(labels) == set(core_methods):
                indices = []
                for label in core_methods:
                    indices.append(labels.index(label))
                handles = [handles[i] for i in indices]
                labels = [labels[i] for i in indices]
                ax.legend(handles, labels, title='Method', frameon=True, ncol=3, bbox_to_anchor=anchor)
            fig.savefig('%s%s_stripplot.pdf' %(args.outprefix, col.replace(' ', '_').replace('\\','')), bbox_inches='tight')
   
        if args.barplot:
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
            sortedgroupedbar(ax, x="Target", y=EFname, groupby="Method", data=allEFs, width=0.7, palette=palette)
            fig.savefig('%sEF%d_targets_barplot.pdf' %(args.outprefix, pctg), bbox_inches='tight')
