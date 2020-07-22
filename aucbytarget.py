#! /usr/bin/env python
import matplotlib as mpl
mpl.use('Agg')
import os,sys,math
import collections
from funcsigs import signature
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from argparse import ArgumentParser
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.lines as mlines

plt.style.use('seaborn-white')
sns.set_palette(sns.color_palette("hls", 8))
mpl.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
mpl.rc('text', usetex=True)

SMALL_SIZE=20
MEDIUM_SIZE=24
BIGGER_SIZE=26
SUBFIG_SIZE=28

plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

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

reverse_map = {'Dense\n(Pose)': 'dense-CNNscore-mean',
        'Dense\n(Affinity)': 'dense-CNNaffinity-mean', 
        'Cross-Docked\n(Pose)': 'crossdock_default2018-CNNscore-mean', 
        'Cross-Docked\n(Affinity)': 'crossdock_default2018-CNNaffinity-mean',
        'General\n(Pose)': 'general_default2018-CNNscore-mean', 
        'General\n(Affinity)': 'general_default2018-CNNaffinity-mean', 
        'RFScore-VS': 'rfscore-vs', 
        'RFScore-4': 'rf-score-4', 
        'Dense\n(Consensus)': 'dense_consensus', 
        'Cross-Docked\n(Consensus)': 'crossdock_default2018_consensus', 
        'General\n(Consensus)': 'general_default2018_consensus'}

swarm_markers = [r'$\clubsuit$', r'$\spadesuit$', '^', '>', '*',
        's', 'o', '<', 'X', 'v', 'h', r'$\P$', '+', '1', '2', '3', '4', 'x']
blue_star = mlines.Line2D([], [], color='black', marker='^', linestyle='None',
                                  markersize=10, label='Blue stars')
red_square = mlines.Line2D([], [], color='red', marker='s', linestyle='None',
                                  markersize=10, label='Red squares')
purple_triangle = mlines.Line2D([], [], color='purple', marker='^',
        linestyle='None',
                                  markersize=10, label='Purple triangles')
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

def mean_auc(data, methods, targets, noskill, args):
        #use this palette if the methods don't correspond to methods used in
        #any of the old papers, which have associated colors
        backup_palette = sns.color_palette("hls", n_colors=len(methods),
                desat=.5).as_hex()

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
        if total_plots > 9:
            mpl.rcParams['xtick.labelsize'] = 10
            mpl.rcParams['ytick.labelsize'] = 10
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
                color = paper_palettes[method] if method in paper_palettes else backup_palette[methods.index(method)]
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
        auc_fig,auc_ax = plt.subplots(figsize=(12.8,9.6))
        aps_fig,aps_ax = plt.subplots()
        if args.make_boxplot:
            if args.color_scheme:
                palette = paper_palettes
            else:
                palette = {}
                for method in methods:
                    palette[method] = paper_palettes[method] if method in \
                        paper_palettes else backup_palette[methods.index(method)]
                # palette = backup_palette
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
                    if marker == '+' or marker == '*':
                        mew = 10
                        size = 15
                    else:
                        mew = 0.5
                        size = 10
                    sns.stripplot(x='Method', y='AUC',
                            data=boxplot_df[boxplot_df['Target']==target],
                            split=True, edgecolor='black', size=size, linewidth=0,
                            linewidths=mew, jitter = True,
                            palette=palette, marker=marker,
                            order=cnnpaper_order, ax=auc_ax)
                    leghands.append(mlines.Line2D([], [], color='black',
                        fillstyle='none', marker=marker, linestyle='None',
                        mew=1,
                        markersize=size, label=target))
                # ax.legend(handles=leghands, bbox_to_anchor=(1,-0.2),
                        # frameon=True)
                auc_ax.legend(handles=leghands, loc='lower left', ncol=2, 
                        frameon=True)
                sns.boxplot(x='Method', y='AUC', data=boxplot_df,
                        color='white', order=cnnpaper_order, ax=auc_ax)
            else:
                # ok actually for now do both if there are few targets
                if len(targets) <= 20:
                    symbol_fig,symbol_ax = plt.subplots()
                    grouped = boxplot_df.groupby(['Method'], as_index=False)
                    medians = grouped['AUC'].median()
                    medians.sort_values(by='AUC', inplace=True)
                    order = medians['Method'].tolist()
                    leghands = []
                    for marker_id,target in enumerate(targets):
                        if marker_id > 11:
                            mew = 2
                            size = 8
                        else:
                            mew = 0.5
                            size = 6
                        marker = swarm_markers[marker_id]
                        sns.stripplot(x='Method', y='AUC',
                                data=boxplot_df[boxplot_df['Target']==target],
                                split=True, size=size,
                                jitter = True, 
                                linewidth=mew,
                                alpha=0.7, 
                                palette=palette, marker=marker,
                                ax=symbol_ax, order=order)
                        leghands.append(mlines.Line2D([], [], color='black',
                            fillstyle='none', marker=marker, linestyle='None',
                            mew=1,
                            markersize=size, label=target))
                    sns.boxplot(x='Method', y='AUC', data=boxplot_df,
                            color='white', ax=symbol_ax, order=order)
                    symbol_ax.legend(handles=leghands, bbox_to_anchor=(1.4, 1.05),
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
            sns.swarmplot(x='Method', y='AUC',
                    data=boxplot_df, split=True, edgecolor='black', size=7,
                    linewidth=0, palette = palette, ax=auc_ax,
                    alpha=0.7, order=order)
            sns.swarmplot(x='Method', y='APS',
                    data=boxplot_df, split=True, edgecolor='black', size=7,
                    linewidth=0, palette = palette, ax=aps_ax)
            sns.boxplot(x='Method', y='AUC', data=boxplot_df,
                    color='white', ax=auc_ax, order=order)
            sns.boxplot(x='Method', y='APS', data=boxplot_df,
                    color='white', ax=aps_ax)
            #sigh
            auc_ax.set_ylabel('AUC')
            auc_ax.set_xlabel('')
            auc_ax.set(ylim=(0,1.1))
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
                    method = contents[4]
                    if method not in ['Vina', 'Vinardo', 'RFScore-VS', 'RFScore-4']:
                        method = name_map[method]
                else:
                    method = contents[4].replace('_', ' ') + modifier
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
        overall_stats = mean_auc(data, methods, targets, noskill, args)
        print(overall_stats)
