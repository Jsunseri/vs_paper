#!/usr/bin/env python
import pickle
from argparse import ArgumentParser
from tabulate import tabulate
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr
import matplotlib as mpl
mpl.use('TKAgg')
import matplotlib.pyplot as plt
import seaborn as sns

from vspaper_settings import paper_palettes, name_map, reverse_map, swarm_markers, litpcba_order

mpl.rcParams.update({'mathtext.fontset': 'cm'})

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
            plot_num = i*2 + j
            sub_ax = plt.subplot2grid((grid_length,grid_width),
                    (plot_num // grid_width, plot_num % grid_width),
                                    fig=fig)
            mean = method + '-' + stype + '-mean'
            stdev = method + '-' + stype + '-std'
            sns.kdeplot(df[mean], df[stdev], 
                    ax=sub_ax,
                    shade=True,
                    shade_lowest=False, 
                    color=paper_palettes[mean])
            r, _ = pearsonr(df[mean].values, df[stdev].values)
            sub_ax.annotate(r'$\rho = {0:.2}$'.format(r), xy=(.1, .9),
                     xycoords=sub_ax.transAxes, family='serif', bbox=props)
            if plot_num == 0:
                sub_ax.set_title('Pose')
            if plot_num == 1:
                sub_ax.set_title('Affinity')
            if j == 0:
                sub_ax.set_ylabel('Ensemble Standard Deviation')
            if i == grid_length-1:
                sub_ax.set_xlabel('Ensemble Mean')
    fig.savefig('ensemble_stdev_vs_mean_allposes.pdf', bbox_inches='tight')

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
            sns.kdeplot(pred_df[mean], pred_df[stdev], 
                    ax=sub_ax,
                    shade=True,
                    shade_lowest=False, 
                    color=paper_palettes[mean])
            r, _ = pearsonr(pred_df[mean].values, pred_df[stdev].values)
            sub_ax.annotate(r'$\rho = {0:.2}$'.format(r), xy=(.1, .9),
                     xycoords=sub_ax.transAxes, family='serif', bbox=props)
            if plot_num == 0:
                sub_ax.set_title('Pose')
            if plot_num == 1:
                sub_ax.set_title('Affinity')
            if j == 0:
                sub_ax.set_ylabel('Ensemble Standard Deviation\n%s' %(name_map[method]))
                sub_ax.set_ylim(0, 0.5)
            else:
                sub_ax.set_ylabel('')
                sub_ax.set_ylim(0, 1.5)
                sub_ax.set_xlim(0, 10)
            # sub_ax.set_xlabel(name_map[mean])
# fig.subplots_adjust(hspace=0.5)
if args.topposes:
    fig.savefig('ensemble_stdev_vs_mean_topposes.pdf', bbox_inches='tight')

targets = pred_df['Target'].unique()
# stdev vs compound minimum rank deviation 
# - do groupby([Target, Title])[Pred].rank(method='first', ascending=False), 
#   which gives their actual rank; then groupby([Target, Title,
#   label])[Pred].rank() to get rank within class. 
# for actives, the rank deviation is the difference classrank - overallrank; 
# for inactives, rank deviation is classrank + nactives - overallrank
if args.rank:
    fig,ax = plt.subplots(figsize=(16,16))
    for i,method in enumerate(cnns):
        for j,stype in enumerate(['CNNscore', 'CNNaffinity']):
            plot_num = i*2 + j
            mname = method + '-' + stype + '-mean'
            pred_df['absrank'] = pred_df.groupby(['Target'])[mname].rank(method='first', ascending=False)
            pred_df['classrank'] = pred_df.groupby(['Target', 'label'])[mname].rank(method='first', ascending=False)
            print(tabulate(pred_df[['label', mname, 'Target', 'Title', 'absrank', 'classrank']], headers='keys', tablefmt='psql'))
    
            tgroups = pred_df.groupby(['Target'], as_index=False)
            actives = tgroups.agg({'label': 'sum'})
            for target in targets:
                pred_df.loc[(pred_df['label'] == 0) & (pred_df['Target'] == target), 'classrank'] = \
                        (pred_df.loc[(pred_df['label'] == 0) & (pred_df['Target'] == target),
                        'classrank']).apply(lambda x: x + actives[actives['Target'] == target]['label'].values[0])
            pred_df['deviation'] = pred_df['classrank'] - pred_df['absrank']
    
            sub_ax = plt.subplot2grid((grid_length,grid_width),
                    (plot_num // grid_width, plot_num % grid_width),
                                    fig=fig)
            stdev = mname.replace('mean','std')
            sns.kdeplot(pred_df['deviation'], pred_df[stdev], 
                    ax = sub_ax,
                    shade=True,
                    shade_lowest=False, 
                    color=paper_palettes[mname])
            r, _ = pearsonr(pred_df[mname].values, pred_df[stdev].values)
            sub_ax.annotate(r'$\rho = {0:.2}$'.format(r), xy=(.1, .9),
                     xycoords=sub_ax.transAxes, family='serif', bbox=props)
            if plot_num == 0:
                sub_ax.set_title('Pose')
            if plot_num == 1:
                sub_ax.set_title('Affinity')
            if j == 0:
                sub_ax.set_ylabel('Ensemble Standard Deviation\n%s' %(name_map[method]))
                sub_ax.set_ylim(0, 0.5)
            else:
                sub_ax.set_ylabel('')
                sub_ax.set_ylim(0, 1.5)
            if i == grid_length-1:
                sub_ax.set_xlabel('Minimum Rank Deviation')
    fig.savefig('ensemble_stdev_vs_rankdeviation_topposes.pdf', bbox_inches='tight')

# per target: mean variance over compounds vs AUC, EF1%
if args.metric:
    auc_fig,auc_ax = plt.subplots(figsize=(16,16))
    ef_fig,ef_ax = plt.subplots(figsize=(16,16))
    grouped = pred_df.groupby(['Target'], as_index=False)
    actives = grouped.agg({'label': 'sum'})
    actives.rename(columns={'label': 'NA'}, inplace=True)
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
       
            R = .01
            pctg = int(R * 100)
            EFname = 'EF{}\%'.format(pctg)
            EFR = actives.merge(enrich_actives, on='Target')
            EFR[EFname] = EFR['na'] / (EFR['NA'] * R)
            EFR = EFR.merge(mean_stdev[stdev, 'Target'], on=['Target'])
            sub_ax = plt.subplot2grid((grid_length,grid_width),
                    (plot_num // grid_width, plot_num % grid_width),
                                    fig=ef_fig)
            sns.scatterplot(EFname, stdev+'-mean', data=EFR, 
                    ax=sub_ax,
                    color=paper_palettes[mname])
            r, _ = pearsonr(EFR[EFname].values, EFR[stdev+'-mean'].values)
            sub_ax.annotate(r'$\rho = {0:.2}$'.format(r), xy=(.1, .9),
                     xycoords=sub_ax.transAxes, family='serif', bbox=props)
            if plot_num == 0:
                sub_ax.set_title('Pose')
            if plot_num == 1:
                sub_ax.set_title('Affinity')
            if j == 0:
                sub_ax.set_ylabel('Ensemble Standard Deviation\nMean Over Each Target\'s Compounds')
            if i == grid_length-1:
                sub_ax.set_xlabel('EF1\%')
    
            # get AUC
            aucs = grouped.apply(lambda x: roc_auc_score(x['label'], x[mname])).rename('AUC').reset_index()
            aucs.merge(mean_stdev, on=['Target'])
            sub_ax = plt.subplot2grid((grid_length,grid_width),
                    (plot_num // grid_width, plot_num % grid_width),
                                    fig=auc_fig)
            sns.scatterplot('AUC', stdev+'-mean', data=aucs, 
                    ax=sub_ax,
                    color=paper_palettes[mname])
            r, _ = pearsonr(aucs['AUC'].values, aucs[stdev+'-mean'].values)
            sub_ax.annotate(r'$\rho = {0:.2}$'.format(r), xy=(.1, .9),
                     xycoords=sub_ax.transAxes, family='serif', bbox=props)
            if plot_num == 0:
                sub_ax.set_title('Pose')
            if plot_num == 1:
                sub_ax.set_title('Affinity')
            if j == 0:
                sub_ax.set_ylabel('Ensemble Standard Deviation\nMean Over Each Target\'s Compounds')
            if i == grid_length-1:
                sub_ax.set_xlabel('AUC')
    
    auc_fig.savefig('ensemble_stdev_vs_auc.pdf', bbox_inches='tight')
    ef_fig.savefig('ensemble_stdev_vs_EF1.pdf', bbox_inches='tight')

# per target: mean stdev over compounds vs max similarity to training set
# read in similarity info
if args.similarity:
    sims = {}
    fig,ax = plt.subplots(figsize=(16,16))
    assert args.simfiles, 'Training set similarity data not provided'
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
    
    # now plot stdev vs that info
    for plot_num,mname in enumerate(means):
        assert method in sims, '%s target similarity data not provided' %method
        # get mean stdev per target
        stdev = mname.replace('mean','std')
        sim_stdev = grouped.agg({stdev: 'mean'})
        sim_stdev.rename(columns={stdev: stdev+'-mean'}, inplace=True)
        sim_stdev['Similarity'] = sim_stdev.apply(lambda x: sims[x[mname]][x['Target']])
        sub_ax = plt.subplot2grid((grid_length,grid_width),
                (plot_num // grid_width, plot_num % grid_width),
                                fig=fig)
        sns.scatterplot('Similarity', stdev+'-mean', data=sim_stdev, 
                ax=sub_ax,
                color=paper_palettes[mname])
        r, _ = pearsonr(sim_stdev['Similarity'].values, sim_stdev[stdev+'-mean'].values)
        sub_ax.annotate(r'$\rho = {0:.2}$'.format(r), xy=(.1, .9),
                 xycoords=sub_ax.transAxes, family='serif', bbox=props)
        if plot_num == 0:
            sub_ax.set_title('Pose')
        if plot_num == 1:
            sub_ax.set_title('Affinity')
        if j == 0:
            sub_ax.set_ylabel('Ensemble Standard Deviation\nMean Over Each Target\'s Compounds')
        if i == grid_length-1:
            sub_ax.set_xlabel('Maximum Similarity to Training Set Target')
    fig.savefig('ensemble_stdev_vs_trainingset_similarity.pdf', bbox_inches='tight')
