#!/usr/bin/env python
import math,pickle
from argparse import ArgumentParser
from tabulate import tabulate

from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import seaborn as sns

from vspaper_settings import (paper_palettes, name_map, reverse_map, 
                              swarm_markers, litpcba_order)

mpl.rcParams.update({'mathtext.fontset': 'cm'})
mpl.rcParams.update({'text.usetex': 'True'})

parser = ArgumentParser(description='Plot relationship between ensemble stdev '
        'and other things that it might be related to')
parser.add_argument('-p', '--pickle', required=True, help='Gotta have a pickle...')
parser.add_argument('-sf', '--simfiles', nargs='+', help='Comma-separated '
        'list of methods/files with target values to indicate training set '
        'similarity to test set targets')
parser.add_argument('-a', '--allposes', action='store_true', help='Do the stdev '
        'vs mean score plot for all poses, not just the top one')
parser.add_argument('-t', '--topposes', action='store_true', help='Do the stdev '
        'vs mean score plot for top pose for each method')
parser.add_argument('-r', '--rank', action='store_true', help='Do the stdev '
        'vs rank deviation plot')
parser.add_argument('-rm', '--rank_vs_mean', action='store_true', help='Do the mean '
        'vs rank deviation plot')
parser.add_argument('-m', '--metric', action='store_true', help='Do the stdev '
        '(mean across all compounds per target) vs performance metric plots')
parser.add_argument('-s', '--similarity', action='store_true', help='Do the '
        'stdev (mean across all compounds per target) vs training set similarity '
        'plots')
args = parser.parse_args()

df = pickle.load(open(args.pickle, 'rb'))
cnns = ["dense", "crossdock_default2018", "general_default2018"]
means = []
for d in cnns:
    for stype in ['CNNscore', 'CNNaffinity']:
        snames = ['%s_seed%d_%s' %(d, seed, stype) for seed in list(range(5))]
        mname = d + '-' + stype + '-mean'
        means.append(mname)
        stdname = d + '-' + stype + '-std'
        df[mname] = df[snames].mean(axis=1)
        df[stdname] = df[snames].std(axis=1)

grid_length = 3
grid_width = 2 
fig,ax = plt.subplots(figsize=(16,16))
props = dict(boxstyle='round', facecolor='white', alpha=0.8)

# variance vs mean, all poses
if args.allposes:
    for i,method in enumerate(cnns):
        for j,stype in enumerate(['CNNscore', 'CNNaffinity']):
            mean = method + '-' + stype + '-mean'
            stdev = method + '-' + stype + '-std'
            print('I found these Inf/NaN values in the DataFrame:\n')
            print(df.index[np.isinf(df[[mean,stdev]]).any(1)])
            print(df.index[np.isnan(df[[mean,stdev]]).any(1)])

            plot_num = i*2 + j
            sub_ax = plt.subplot2grid((grid_length,grid_width),
                    (plot_num // grid_width, plot_num % grid_width),
                                    fig=fig)
            sns.scatterplot(df[mean], df[stdev], 
                    ax=sub_ax,
                    # shade=True,
                    # shade_lowest=False, 
                    linewidths=0.5,
                    edgecolor='black',
                    alpha=0.5,
                    color=paper_palettes[mean])
            # r, _ = spearmanr(df[mean].values, df[stdev].values)
            # sub_ax.annotate(r'$\rho = {0:.2}$'.format(r), xy=(.1, .9),
                     # xycoords=sub_ax.transAxes, family='serif', bbox=props)
            sub_ax.set_xlabel('')
            if plot_num == 0:
                sub_ax.set_title('Pose')
            if plot_num == 1:
                sub_ax.set_title('Affinity')
            if j == 0:
                sub_ax.set_ylabel('Ensemble Standard Deviation\n%s' %(name_map[method]))
                sub_ax.set_ylim(sub_ax.get_ylim()[0], 0.56)
            else:
                sub_ax.set_ylabel('')
                sub_ax.set_ylim(sub_ax.get_ylim()[0], 2.55)
                sub_ax.set_xlim(0, 10.5)
    fig.savefig('ensemble_stdev_vs_mean_allposes.png', dpi=300, bbox_inches='tight')

grouped = df.groupby(['Target','Title'], as_index=False)
fig,ax = plt.subplots(figsize=(16,16))
# variance vs mean, actual prediction
for i,method in enumerate(cnns):
    for j,stype in enumerate(['CNNscore', 'CNNaffinity']):
        mean = method + '-' + stype + '-mean'
        stdev = method + '-' + stype + '-std'
        this_df = df[['label', 'Target', 'Title', mean, stdev]].loc[grouped[mean].idxmax()]
        try:
            pred_df = pred_df.merge(this_df, on=['label', 'Target', 'Title'], sort=False)
        except NameError:
            pred_df = this_df
        if args.topposes:
            plot_num = i*2 + j
            sub_ax = plt.subplot2grid((grid_length,grid_width),
                    (plot_num // grid_width, plot_num % grid_width),
                                    fig=fig)
            sns.scatterplot(pred_df[mean], pred_df[stdev], 
                    ax=sub_ax,
                    # shade=True,
                    # shade_lowest=False, 
                    linewidths=0.5,
                    edgecolor='black',
                    alpha=0.5,
                    color=paper_palettes[mean])
            # r, _ = spearmanr(pred_df[mean].values, pred_df[stdev].values)
            # sub_ax.annotate(r'$\rho = {0:.2}$'.format(r), xy=(.1, .9),
                     # xycoords=sub_ax.transAxes, family='serif', bbox=props)
            sub_ax.set_xlabel('')
            if plot_num == 0:
                sub_ax.set_title('Pose')
            if plot_num == 1:
                sub_ax.set_title('Affinity')
            if j == 0:
                sub_ax.set_ylabel('Ensemble Standard Deviation\n%s' %(name_map[method]))
                sub_ax.set_ylim(sub_ax.get_ylim()[0], 0.56)
            else:
                sub_ax.set_ylabel('')
                sub_ax.set_ylim(sub_ax.get_ylim()[0], 2.05)
                sub_ax.set_xlim(0, 10.5)
            # sub_ax.set_xlabel(name_map[mean])
# fig.subplots_adjust(hspace=0.5)
if args.topposes:
    fig.savefig('ensemble_stdev_vs_mean_topposes.png', dpi=300, bbox_inches='tight')

targets = pred_df['Target'].unique()
# stdev vs compound minimum rank deviation 
# - do groupby([Target, Title])[Pred].rank(method='first', ascending=False), 
#   which gives their actual rank; then groupby([Target, Title,
#   label])[Pred].rank() to get rank within class. 
# for actives, the rank deviation is the difference classrank - overallrank; 
# for inactives, rank deviation is classrank + nactives - overallrank
if args.rank:
    ctypes = ['active', 'inactive']
    figinfo = {}
    for ctype in ctypes:
        figinfo[ctype] = plt.subplots(figsize=(16,16))
    for i,method in enumerate(cnns):
        for j,stype in enumerate(['CNNscore', 'CNNaffinity']):
            plot_num = i*2 + j
            mname = method + '-' + stype + '-mean'
            pred_df['absrank'] = pred_df.groupby(['Target'])[mname].rank(method='first', ascending=False)
            pred_df['classrank'] = pred_df.groupby(['Target', 'label'])[mname].rank(method='first', ascending=False)
            tgroups = pred_df.groupby(['Target'], as_index=False)
            actives = tgroups.agg({'label': 'sum'})
            for target in targets:
                pred_df.loc[(pred_df['label'] == 0) & (pred_df['Target'] == target), 'classrank'] = \
                        (pred_df.loc[(pred_df['label'] == 0) & (pred_df['Target'] == target),
                        'classrank']).apply(lambda x: x + actives[actives['Target'] == target]['label'].values[0])
            pred_df['deviation'] = pred_df['classrank'] - pred_df['absrank']
            # print(tabulate(pred_df[['label', mname, 'Target', 'Title',
                # 'absrank', 'classrank', 'deviation']], headers='keys', tablefmt='psql'))
   
            # two separate figs because the magnitude of the negative deviations 
            # associated with actives hides the detail of what's happening with
            # the inactives
            for ctype in ctypes:
                label = 0 if ctype == 'inactive' else 1
                sub_df = pred_df.loc[pred_df['label'] == label]
                fig,ax = figinfo[ctype]
                sub_ax = plt.subplot2grid((grid_length,grid_width),
                        (plot_num // grid_width, plot_num % grid_width),
                                        fig=fig)
                stdev = mname.replace('mean','std')
                sub_ax.hexbin(sub_df['deviation'], sub_df[stdev], mincnt=1, 
                        gridsize=50, color=paper_palettes[mname])
                r, _ = spearmanr(sub_df[mname].values, sub_df[stdev].values)
                sub_ax.annotate(r'$\rho = {0:.2}$'.format(r), xy=(.1, .9),
                         xycoords=sub_ax.transAxes, family='serif', bbox=props)
                sub_ax.set_xlabel('')
                if plot_num == 0:
                    sub_ax.set_title('Pose')
                if plot_num == 1:
                    sub_ax.set_title('Affinity')
                if j == 0:
                    sub_ax.set_ylabel('Ensemble Standard Deviation\n%s' %(name_map[method]))
                    sub_ax.set_ylim(sub_ax.get_ylim()[0], 0.56)
                else:
                    sub_ax.set_ylabel('')
                    sub_ax.set_ylim(sub_ax.get_ylim()[0], 2.05)
                if i == grid_length-1:
                    sub_ax.set_xlabel('Minimum Rank Deviation')
                # sub_ax.set_xticklabels([i.get_text().replace('−', '$-$') for i in sub_ax.get_xticklabels()])
    for ctype in ctypes:
        fig,_ = figinfo[ctype]
        fig.savefig('ensemble_stdev_vs_rankdeviation_topposes_hex_%s.png' %ctype, dpi=300, bbox_inches='tight')

if args.rank_vs_mean:
    ctypes = ['active', 'inactive']
    figinfo = {}
    for ctype in ctypes:
        figinfo[ctype] = plt.subplots(figsize=(16,16))
    for i,method in enumerate(cnns):
        for j,stype in enumerate(['CNNscore', 'CNNaffinity']):
            plot_num = i*2 + j
            mname = method + '-' + stype + '-mean'
            pred_df['absrank'] = pred_df.groupby(['Target'])[mname].rank(method='first', ascending=False)
            pred_df['classrank'] = pred_df.groupby(['Target', 'label'])[mname].rank(method='first', ascending=False)
            tgroups = pred_df.groupby(['Target'], as_index=False)
            actives = tgroups.agg({'label': 'sum'})
            for target in targets:
                pred_df.loc[(pred_df['label'] == 0) & (pred_df['Target'] == target), 'classrank'] = \
                        (pred_df.loc[(pred_df['label'] == 0) & (pred_df['Target'] == target),
                        'classrank']).apply(lambda x: x + actives[actives['Target'] == target]['label'].values[0])
            pred_df['deviation'] = pred_df['classrank'] - pred_df['absrank']
   
            # two separate figs because the magnitude of the negative deviations 
            # associated with actives hides the detail of what's happening with
            # the inactives
            for ctype in ctypes:
                label = 0 if ctype == 'inactive' else 1
                sub_df = pred_df.loc[pred_df['label'] == label]
                fig,ax = figinfo[ctype]
                sub_ax = plt.subplot2grid((grid_length,grid_width),
                        (plot_num // grid_width, plot_num % grid_width),
                                        fig=fig)
                sns.scatterplot(sub_df['deviation'], sub_df[mname], 
                        ax = sub_ax,
                        linewidths=0.5,
                        edgecolor='black',
                        alpha=0.7,
                        # shade=True,
                        # shade_lowest=False, 
                        color=paper_palettes[mname])
                sub_ax.set_xlabel('')
                if plot_num == 0:
                    sub_ax.set_title('Pose')
                if plot_num == 1:
                    sub_ax.set_title('Affinity')
                if j == 0:
                    sub_ax.set_ylabel('Ensemble Mean\n%s' %(name_map[method]))
                else:
                    sub_ax.set_ylabel('')
                    sub_ax.set_ylim(sub_ax.get_ylim()[0], 10.5)
                if i == grid_length-1:
                    sub_ax.set_xlabel('Minimum Rank Deviation')
    for ctype in ctypes:
        fig,_ = figinfo[ctype]
        fig.savefig('ensemble_mean_vs_rankdeviation_topposes_%s.png' %ctype, dpi=300, bbox_inches='tight')

# per target: mean variance over compounds vs AUC, EF1%
if args.metric:
    auc_fig,auc_ax = plt.subplots(figsize=(16,16))
    ef_fig,ef_ax = plt.subplots(figsize=(16,16))
    grouped = pred_df.groupby(['Target'], as_index=False)
    actives = grouped.agg({'label': 'sum'})
    actives.rename(columns={'label': 'NA'}, inplace=True)
    R = .01
    pctg = int(R * 100)
    EFname = 'EF{}\%'.format(pctg)
    for i,method in enumerate(cnns):
        for j,stype in enumerate(['CNNscore', 'CNNaffinity']):
            plot_num = i*2 + j
            mname = method + '-' + stype + '-mean'
            # get mean stdev per target
            stdev = mname.replace('mean','std')
            mean_stdev = grouped.agg({stdev: 'mean'})
            mean_stdev.rename(columns={stdev: stdev+'-mean'}, inplace=True)
    
            # get EF1%
            topn = grouped.apply(lambda x: x.nlargest(math.ceil(len(x) * R), mname))
            topn.reset_index(inplace=True, drop=True)
            topn_grouped = topn.groupby(['Target'], as_index=False)
            enrich_actives = topn_grouped.agg({'label': 'sum'})
            enrich_actives.rename(columns={'label': 'na'}, inplace=True)
       
            EFR = actives.merge(enrich_actives, on='Target')
            EFR[EFname] = EFR['na'] / (EFR['NA'] * R)
            EFR = EFR.merge(mean_stdev[[stdev+'-mean', 'Target']], on=['Target'])
            sub_ax = plt.subplot2grid((grid_length,grid_width),
                    (plot_num // grid_width, plot_num % grid_width),
                                    fig=ef_fig)
            sns.scatterplot(EFname, stdev+'-mean', data=EFR, 
                    ax=sub_ax,
                    linewidths=0.5,
                    edgecolor='black',
                    alpha=0.7,
                    color=paper_palettes[mname])
            # r, _ = spearmanr(EFR[EFname].values, EFR[stdev+'-mean'].values)
            # sub_ax.annotate(r'$\rho = {0:.2}$'.format(r), xy=(.1, .9),
                     # xycoords=sub_ax.transAxes, family='serif', bbox=props)
            sub_ax.set_xlabel('')
            if plot_num == 0:
                sub_ax.set_title('Pose')
            if plot_num == 1:
                sub_ax.set_title('Affinity')
            if j == 0:
                sub_ax.set_ylabel('Ensemble Standard Deviation')
                sub_ax.set_ylim(0, 0.3)
            else:
                sub_ax.set_ylabel('')
                sub_ax.set_ylim(0, 0.75)
            if i == grid_length-1:
                sub_ax.set_xlabel('EF1%')
            sub_ax.set_xlim(sub_ax.get_xlim()[0], 80)
    
            # get AUC
            idx_grouped = pred_df.groupby(['Target'])
            aucs = idx_grouped.apply(lambda x: roc_auc_score(x['label'], x[mname])).rename('AUC').reset_index()
            aucs = aucs.merge(mean_stdev[[stdev+'-mean', 'Target']], on=['Target'])
            sub_ax = plt.subplot2grid((grid_length,grid_width),
                    (plot_num // grid_width, plot_num % grid_width),
                                    fig=auc_fig)
            sns.scatterplot('AUC', stdev+'-mean', data=aucs, 
                    ax=sub_ax,
                    linewidths=0.5,
                    edgecolor='black',
                    alpha=0.7,
                    color=paper_palettes[mname])
            # r, _ = spearmanr(aucs['AUC'].values, aucs[stdev+'-mean'].values)
            # sub_ax.annotate(r'$\rho = {0:.2}$'.format(r), xy=(.1, .9),
                     # xycoords=sub_ax.transAxes, family='serif', bbox=props)
            sub_ax.set_xlabel('')
            if plot_num == 0:
                sub_ax.set_title('Pose')
            if plot_num == 1:
                sub_ax.set_title('Affinity')
            if j == 0:
                sub_ax.set_ylabel('Ensemble Standard Deviation')
                sub_ax.set_ylim(0, 0.3)
            else:
                sub_ax.set_ylabel('')
                sub_ax.set_ylim(0, 0.75)
            if i == grid_length-1:
                sub_ax.set_xlabel('AUC')
    
    auc_fig.savefig('ensemble_stdev_vs_auc.png', dpi=300, bbox_inches='tight')
    ef_fig.savefig('ensemble_stdev_vs_EF1.png', dpi=300, bbox_inches='tight')

# per target: mean stdev over compounds vs max similarity to training set
# read in similarity info
if args.similarity:
    sims = {}
    fig,ax = plt.subplots(figsize=(16,16))
    assert args.simfiles, 'Training set similarity data not provided'
    grouped = pred_df.groupby(['Target'], as_index=False)
    for sim in args.simfiles:
        siminfo = sim.strip().split(',')
        sim_method = siminfo[0]
        simfile = siminfo[1]
        sims[sim_method] = {}
        # read file contents into dict
        with open(simfile, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                contents = line.strip().split()
                target = contents[0]
                similarity = 1 - float(contents[-1])
                sims[sim_method][target] = similarity
                # target = contents[0].replace('_', ' ')
        for target in targets:
            assert target in sims[sim_method], 'Target %s missing from similarity data %s' %(target, simfile)
    
    # now plot stdev vs that info
    for i,method in enumerate(cnns):
        for j,stype in enumerate(['CNNscore', 'CNNaffinity']):
            assert method in sims, '%s target similarity data not provided' %method
            plot_num = i*2 + j
            # get mean stdev per target
            mname = method + '-' + stype + '-mean'
            stdev = mname.replace('mean','std')
            sim_stdev = grouped.agg({stdev: 'mean'})
            sim_stdev.rename(columns={stdev: stdev+'-mean'}, inplace=True)
            sim_stdev['Similarity'] = sim_stdev.apply(lambda x: sims[method][x['Target']], axis=1)
            sub_ax = plt.subplot2grid((grid_length,grid_width),
                    (plot_num // grid_width, plot_num % grid_width),
                                    fig=fig)
            sns.scatterplot('Similarity', stdev+'-mean', data=sim_stdev, 
                    ax=sub_ax,
                    linewidths=0.5,
                    edgecolor='black',
                    alpha=0.7,
                    color=paper_palettes[mname])
            # r, _ = spearmanr(sim_stdev['Similarity'].values, sim_stdev[stdev+'-mean'].values)
            # sub_ax.annotate(r'$\rho = {0:.2}$'.format(r), xy=(.1, .9),
                     # xycoords=sub_ax.transAxes, family='serif', bbox=props)
            sub_ax.set_xlabel('')
            if plot_num == 0:
                sub_ax.set_title('Pose')
            if plot_num == 1:
                sub_ax.set_title('Affinity')
            if j == 0:
                sub_ax.set_ylabel('Ensemble Standard Deviation')
                sub_ax.set_ylim(sub_ax.get_ylim()[0], 0.5)
            else:
                sub_ax.set_ylabel('')
                sub_ax.set_ylim(sub_ax.get_ylim()[0], 1.5)
            if i == grid_length-1:
                sub_ax.set_xlabel('Maximum Similarity to Training Set Target')
    fig.savefig('ensemble_stdev_vs_trainingset_similarity.png', dpi=300, bbox_inches='tight')
