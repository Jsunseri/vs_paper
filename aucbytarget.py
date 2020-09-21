#! /usr/bin/env python
import os,sys,math
import collections
from argparse import ArgumentParser
from funcsigs import signature

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

from vspaper_settings import paper_palettes, backup_palette, name_map, reverse_map, swarm_markers, litpcba_successes, litpcba_order, marker_sizes

# In matplotlib < 1.5, plt.fill_between does not have a 'step'
# argument
step_kwargs = ({'step': 'post'}
                if 'step' in
                signature(plt.fill_between).parameters
                else {})

def calc_auc_and_pr(target_and_method, target_predictions):
        y_true=[]
        y_score=[]
        for i,item in enumerate(target_predictions):
            try:
                label = float(item[0])
                score = float(item[1])
                y_true.append(label)
                y_score.append(score)
            except Exception as e:
                print('Error: %d %f %s\n'%(label, score, target_and_method[i]))
                continue
        fpr,tpr,_ = roc_curve(y_true,y_score)
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        return {'auc': (roc_auc_score(y_true,y_score),fpr,tpr), 'aupr' :
                (average_precision_score(y_true, y_score),
                    precision, recall)}

def mean_auc(data, methods, targets, noskill, sims, args):
        #overall_auc tracks and reports the average AUC across all targets
        #bytarget tracks and reports the AUC for each target for each method
        overall_stats = {}
        bytarget = {}
        for method in methods:
            if method in reverse_map:
                out_method = reverse_map[method]
                bytarget[method] = open('%s_%s'%(out_method,args.outprefix),'w')
            else:
                bytarget[method] = open('%s_%s'%(method,args.outprefix),'w')
        if args.make_boxplot:
            boxplot_dat = []
        total_plots = len(targets)
        # if total_plots > 9:
            # mpl.rcParams['xtick.labelsize'] = 10
            # mpl.rcParams['ytick.labelsize'] = 10
        grid_width = int(math.ceil(math.sqrt(total_plots)))
        grid_length = int(math.ceil(float(total_plots)/grid_width))
        fig,ax = plt.subplots(figsize=(16,16))

        #if there is only one output plot, print AUCs on the legend; otherwise
        #make a single shared legend for all plots without AUCs
        #everything is duplicated for APS now
        if args.prec_rec:
            chosen_stat = "APS"
        else:
            chosen_stat = "AUC"

        legend_dict = {}
        for output_stat in ['AUC', 'APS']:
            legend_dict[output_stat] = {}
            overall_stats[output_stat] = {}
        num_lines = {}
        for t,l in data.items():
                summary_stats = calc_auc_and_pr(t,l)
                auc,fpr,tpr = summary_stats['auc']
                aps,precision,recall = summary_stats['aupr']
                bytarget[t[1]].write('%s %.3f %.3f\n' %(t[0].replace(' ', '_'),auc,aps))
                if args.make_boxplot:
                    boxplot_dat.append({'Method' : t[1], 'AUC' : auc, 'Target'
                        : t[0], 'APS': aps})
                    if sims:
                        boxplot_dat[-1]['Similarity'] = sims[t[1]][t[0]]
                plot_num = targets.index(t[0])
                if t[1] not in overall_stats['AUC']:
                    overall_stats['AUC'][t[1]] = 0
                    overall_stats['APS'][t[1]] = 0
                overall_stats['AUC'][t[1]] += float(auc)
                overall_stats['APS'][t[1]] += float(aps)
                if plot_num not in legend_dict[chosen_stat]: 
                    num_lines[plot_num] = 0 
                    sub_ax = plt.subplot2grid((grid_length,grid_width),
                            (plot_num // grid_width, plot_num % grid_width),
                            fig=fig)
                    sub_ax.set_aspect('equal')
                    # if not args.prec_rec:
                        # for tick in sub_ax.get_xticklabels():
                            # tick.set_rotation(-90)
                    legend_dict[chosen_stat][plot_num] = sub_ax
                method = t[1]
                color = paper_palettes[method] if method in paper_palettes else \
                        backup_palette[methods.index(method) % len(backup_palette)]
                if chosen_stat == "AUC":
                    label = t[1]
                    legend_dict['AUC'][plot_num].plot(fpr, tpr, color=color,
                            label=label, lw=5, zorder=2) 
                    legend_dict['AUC'][plot_num].set(ylim=(0.0, 1.0))
                    legend_dict['AUC'][plot_num].set(xlim=(0.0, 1.0))
                else:
                    label = '%s, APS=%0.2f' % (t[1], aps) if total_plots == 1 else t[1]
                    legend_dict['APS'][plot_num].step(recall, precision, alpha=0.2,
                                     where='post', label=label, color=color)
                    legend_dict['APS'][plot_num].fill_between(recall,
                            precision, alpha=0.2, color=color,
                            **step_kwargs)
                    legend_dict['APS'][plot_num].set(ylim=(0.0, 1.05))
                    legend_dict['APS'][plot_num].set(xlim=(0.0, 1.0))

                num_lines[plot_num] += 1
                if int(plot_num) / grid_width == grid_length-1:
                    if chosen_stat == 'AUC':
                        legend_dict['AUC'][plot_num].set_xlabel('FPR')
                    else:
                        legend_dict['APS'][plot_num].set_xlabel('Recall')
                if plot_num % grid_width == 0:
                    if chosen_stat == 'AUC':
                        legend_dict['AUC'][plot_num].set_ylabel('TPR')
                    else:
                        legend_dict['APS'][plot_num].set_ylabel('Precision')
                if num_lines[plot_num] == len(methods):
                    #add in line showing random performance
                    if chosen_stat == 'AUC':
                        locs = legend_dict['AUC'][plot_num].get_xticks()
                        legend_dict['AUC'][plot_num].set_xticks(locs[1:])
                        legend_dict['AUC'][plot_num].plot([0, 1], [0, 1],
                                color='gray', lw=5,
                           linestyle='--', zorder=1)
                    else:
                        target_noskill = noskill[t[0]][0]/float(noskill[t[0]][1])
                        legend_dict['APS'][plot_num].plot([0, 1],
                                [target_noskill, target_noskill], color='gray',
                                lw=2,
                           linestyle='--', zorder=1)

        #if we have multiple subplots, make a shared legend; constrain CNN
        #paper order to follow line order
        #these papers didn't have PR curves so i'm not touching it for them
        if chosen_stat == 'AUC':
            handles,labels = legend_dict['AUC'][0].get_legend_handles_labels()
        else:
            handles,labels = legend_dict['APS'][0].get_legend_handles_labels()
        if args.color_scheme == 'cnn':
            shortlabels = [x.split(',')[0] for x in labels]
            indices = []
            cnnpaper_order = ['DUD-E', 'Vina', '2:1', 'CSAR']
            for label in cnnpaper_order:
                indices.append(shortlabels.index(label))
            handles = [handles[i] for i in indices]
            labels = [labels[i] for i in indices]
            box = legend_dict['AUC'][total_plots-2].get_position()
            fig.legend(handles, labels, loc=(box.x0+0.465, box.y0-0.035), frameon=True)
        elif args.color_scheme == 'd3r':
            indices = []
            cnnpaper_order = ['CNN Affinity Rescore', 'CNN Affinity Refine',
                    'CNN Scoring Rescore', 'CNN Scoring Refine', 'Vina']
            for label in cnnpaper_order:
                indices.append(labels.index(label))
            handles = [handles[i] for i in indices]
            labels = [labels[i] for i in indices]
            fig.legend(handles, labels, loc='lower center', ncol=3, frameon=True)
        elif args.color_scheme == 'overlap':
            indices = []
            # overlap_order = ['Vina', 'CNN','CNNscore', 'CNNaffinity', 'Overlap L2', 'Overlap Mult']
            overlap_order = ['Overlap Mult', 'Overlap L2', 'CNN','CNNscore', 'CNNaffinity', 'Vina']
            for label in overlap_order:
                if label in labels:
                    indices.append(labels.index(label))
            handles = [handles[i] for i in indices]
            labels = [labels[i] for i in indices]
            fig.legend(handles, labels, loc='upper center', ncol=2, frameon=True)
        else:
            legend = fig.legend(handles, labels, loc='best', frameon=True)

        if args.color_scheme == 'cnn':
            fig.subplots_adjust(wspace=0.05, hspace=0.5)
        else:
            #TODO: currently doing this via manual tuning - can it be
            #automated? if you are using this, you may need to fiddle with
            #these numbers and be aware that if you try to use tight_layout it
            #seems to override whatever you do here
            fig.subplots_adjust(hspace=0.6, wspace=0.55)
        for method in overall_stats['AUC']:
            overall_stats['AUC'][method] /= total_plots
            overall_stats['APS'][method] /= total_plots
            # outfile.write('%s, AUC %.2f\n' 
                    # % (method.split('_')[0],overall_auc[method]))
            bytarget[method].close()
        fig.savefig(args.outprefix+'_%s.pdf'%chosen_stat, bbox_inches='tight')

        #now do boxplots
        auc_fig,auc_ax = plt.subplots(figsize=(12,10))
        aps_fig,aps_ax = plt.subplots()
        if args.make_boxplot:
            palette = {}
            for method in methods:
                palette[method] = paper_palettes[method] if method in \
                    paper_palettes else backup_palette[methods.index(method) % len(backup_palette)]
            boxplot_df = pd.DataFrame(boxplot_dat)
            #if we're doing the d3r paper figs, David wants a different marker
            #style for each target because having per-target ROC plots isn't
            #enough for him...
            if args.color_scheme == 'd3r':
                cnnpaper_order = ['CNN Affinity Rescore', 'CNN Affinity Refine',
                        'CNN Scoring Rescore', 'CNN Scoring Refine', 'Vina']
                leghands = []
                for marker_id,target in enumerate(targets):
                    marker = swarm_markers[marker_id]
                    mew = 0.5
                    if marker in marker_sizes:
                        size = marker_sizes[marker]
                    else:
                        size = 12
                    sns.stripplot(x='Method', y='AUC',
                            data=boxplot_df[boxplot_df['Target']==target],
                            split=True, edgecolor='black', size=size, linewidth=0,
                            linewidths=mew, jitter = True,
                            palette=palette, marker=marker,
                            order=cnnpaper_order, ax=auc_ax)
                    leghands.append(mlines.Line2D([], [], color='black',
                        fillstyle='none', marker=marker, linestyle='None',
                        mew=0.5,
                        markersize=size, label=target))
                # ax.legend(handles=leghands, bbox_to_anchor=(1,-0.2),
                        # frameon=True)
                auc_ax.legend(handles=leghands, loc='lower left', ncol=2, 
                        frameon=True)
                sns.boxplot(x='Method', y='AUC', data=boxplot_df,
                        color='white', order=cnnpaper_order, ax=auc_ax, 
                        showfliers=False)
            else:
                # ok actually for now do both if there are few targets
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
                                if (elem['Method'] == m) and (round(elem['AUC'],3) == round(auc,3)):
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
            grouped = boxplot_df.groupby(['Method'], as_index=False)
            medians = grouped['AUC'].median()
            medians.sort_values(by='AUC', inplace=True)
            order = medians['Method'].tolist()
            sns.boxplot(x='Method', y='AUC', data=boxplot_df,
                    color='white', ax=auc_ax, order=order, 
                    showfliers=False)
            sns.boxplot(x='Method', y='APS', data=boxplot_df,
                    color='white', ax=aps_ax, 
                    showfliers=False)
            if sims:
                # evidently we can't pass a list/array of arrays as y and select specific x, 
		# so instead we're going to set any excluded methods to a dummy
                # value and then fix the axis limits so the dummy value is cropped out
                # the boxplot uses the real data so it's unaffected by the off-plot dummy point
                threshold = 0.7
                relevant = boxplot_df.loc[boxplot_df.Similarity < threshold]
                labels = relevant['Method'].unique().tolist()
                dummy = {}
                dummy['Method'] = []
                dummy['AUC'] = []
                for mn in order:
                    if mn not in labels:
                        dummy['Method'].append(mn)
                        dummy['AUC'].append(-1)
                if dummy['Method']:
                    relevant = relevant.append(pd.DataFrame(dummy))
                sns.swarmplot(x='Method', y='AUC', 
                        data=relevant, order=order, 
                        palette = palette, 
                        split=True, 
                        edgecolor='black', size=7, marker='o', 
                        linewidth=0, ax=auc_ax,
                        alpha=0.7)
                relevant = boxplot_df.loc[boxplot_df.Similarity > threshold]
                labels = relevant['Method'].unique().tolist()
                dummy = {}
                dummy['Method'] = []
                dummy['AUC'] = []
                for mn in order:
                    if mn not in labels:
                        dummy['Method'].append(mn)
                        dummy['AUC'].append(-1)
                if dummy['Method']:
                    relevant = relevant.append(pd.DataFrame(dummy))
                sns.swarmplot(x='Method', y='AUC', 
                        data=relevant, order=order, 
                        palette = palette, 
                        split=True, 
                        edgecolor='black', size=7, marker='X', 
                        linewidth=0, ax=auc_ax,
                        alpha=0.7)
            else:
                sns.swarmplot(x='Method', y='AUC',
                        data=boxplot_df, split=True, edgecolor='black', size=7,
                        linewidth=0, palette = palette, ax=auc_ax,
                        alpha=0.7, order=order)
            sns.swarmplot(x='Method', y='APS',
                    data=boxplot_df, split=True, edgecolor='black', size=7,
                    linewidth=0, palette = palette, ax=aps_ax)
            #sigh
            auc_ax.set_ylabel('AUC')
            auc_ax.set_xlabel('')
            auc_ax.set(ylim=(0,1.05))
            auc_xlims = auc_ax.get_xlim()
            auc_ax.plot([auc_xlims[0],auc_xlims[1]],[0.5, 0.5], linestyle='--', color='gray',
                    zorder=1, alpha=0.5)
            if not args.color_scheme == 'overlap' and not args.color_scheme == 'vspaper':
                for tick in auc_ax.get_xticklabels():
                    tick.set_rotation(90)
            #APS
            aps_ax.set_ylabel('APS')
            aps_ax.set_xlabel('')
            aps_ax.set(ylim=(0,1.1))
            aps_xlims = auc_ax.get_xlim()
            for tick in aps_ax.get_xticklabels():
                tick.set_rotation(45)
            #special cases for d3r paper
            if args.color_scheme == 'd3r':
                labels = auc_ax.get_xticklabels()
                labels = [label.get_text().split() for label in labels]
                labels = ['%s %s\n%s'%(x[0],x[1],x[2]) if not x[0] == 'Vina' else x[0] for x in labels]
                auc_ax.set_xticklabels(labels)
            auc_fig.savefig(args.outprefix+'_auc_boxplot.pdf', bbox_inches='tight')
            aps_fig.savefig(args.outprefix+'_aps_boxplot.pdf', bbox_inches='tight')
        return overall_stats

if __name__=='__main__':
        parser = ArgumentParser(description='Calculate AUC/AUPR by target for multiple methods')
        parser.add_argument('-p','--predictions', nargs='*', default=[], 
                help='files of predictions, formatted LABELS PREDICTIONS TARGET METHOD where METHOD is stylized as desired for output, with spaces replaced with underscores')
        parser.add_argument('-o','--outprefix',type=str,default='bytarget',help='prefix for all output files')
        parser.add_argument('-make_boxplot', action='store_true',
                default=False, help='Make a boxplot of the by-target values \
associated with each method')
        parser.add_argument('-pr', '--prec_rec', default=False,
                action='store_true', help='Plot AUPR instead of AUC.')
        parser.add_argument('-color_scheme', required=False, choices=['cnn',
            'd3r', 'overlap', 'vspaper'], 
            help='Specify color scheme, options are cnn, d3r, overlap, or vspaper; if used, the \
predictions files must have names indicating the correct methods to \
use those color schemes')
        parser.add_argument('-s', '--simfiles', nargs='*', default=[], help='Optionally '
                'provide a comma-separated list of methods/files with target values to define marker '
                'symbols based on a threshold value (originally intended to '
                'represent similarity to training set)')
        args= parser.parse_args()

        data = {}
        noskill = {}
        methods,targets = [],[]
        if args.color_scheme == 'cnn':
            paper_palettes['Vina'] = '#CCBB44'
        for file in args.predictions:
            modifier = ''
            # if "scorerank" not in file:
                # modifier = "-affinityrank"
            # if "max" not in file:
                # modifier += "-worstpose"
            for line in open(file,'r'):
                contents = line.split()
                target = contents[2].replace('_', ' ')
                if args.color_scheme == 'vspaper':
                    method = contents[-1]
                    if method in name_map:
                        method = name_map[method]
                    else:
                        method = method.replace('_', ' ')
                else:
                    method = contents[-1].replace('_', ' ') + modifier
                if target not in targets:
                    targets.append(target)
                    noskill[target] = [0,0]
                if method not in methods:
                    methods.append(method)
                this_key = (target,method)
                if this_key not in data:
                    data[this_key] = []
                data[this_key].append((contents[0],contents[1]))
                noskill[target][0] += (int(float(contents[0])) == 1)
                noskill[target][1] += 1
        sims = {}
        for sim in args.simfiles:
            siminfo = sim.strip().split(',')
            sim_method = siminfo[0]
            if sim_method in name_map:
                sim_method = name_map[sim_method]
            simfile = siminfo[1]
            sims[sim_method] = {}
            # read file contents into dict
            with open(simfile, 'r') as f:
                for line in f:
                    if line.startswith('#'):
                        continue
                    contents = line.strip().split()
                    target = contents[0].replace('_', ' ')
                    similarity = 1 - float(contents[-1])
                    sims[sim_method][target] = similarity
            for target in targets:
                assert target in sims[sim_method], 'Target %s missing from similarity data %s' %(target, simfile)
        if args.simfiles:
            for method in methods:
                if method not in sims:
                    sims[method] = {}
                    for target in targets:
                        # assume there was no training data
                        sims[method][target] = 0
        if len(set(targets).intersection(litpcba_order)) == len(targets):
            targets = [t for t in litpcba_order if t in targets]
        overall_stats = mean_auc(data, methods, targets, noskill, sims, args)
        print(overall_stats)
