#!/usr/bin/env python
import glob, sys, os, argparse,math

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.gridspec as gridspec

from scipy.stats import spearmanr

import seaborn as sns
import pandas as pd
from tabulate import tabulate

from vspaper_settings import (paper_palettes, backup_palette, name_map, reverse_map, 
			      swarm_markers, litpcba_ntemplates, litpcba_order, marker_sizes,
			      SeabornFig2Grid)
from early_enrichment import getEF

parser = argparse.ArgumentParser(description='Plot change in performance '
        'per method when taking max vs min pose as the basis of the prediction')
parser.add_argument('--max', nargs='+', help='Files summarizing '
        'performance using max-scoring pose per (target,compound) pair')
parser.add_argument('--min', nargs='+', help='Files summarizing '
        'performance using min-scoring pose per (target,compound) pair')
args = parser.parse_args()

paper_methods = ['Dense\n(Pose)', 'Dense\n(Affinity)', 
		 'Cross-Docked\n(Pose)', 'Cross-Docked\n(Affinity)',
 		 'General\n(Pose)', 'General\n(Affinity)',
                 'Vina', 'Vinardo', 'RFScore-4', 'RFScore-VS']

black_circle = mlines.Line2D([], [], color='black', marker='o', linestyle='None',
                                  markersize=16, label='Black Circles')
white_circle = mlines.Line2D([], [], color='white', marker='o', linestyle='None',
                                      mec='black', mew=0.5, markersize=16, label='White Circles')

R = 0.01
pctg = int(R * 100)
EFname = 'EF{}%'.format(pctg) 

cols = ['Label', 'Prediction', 'Target', 'Title', 'Method']
max_dfs = {}
# read in max files
for f in args.max:
    this_df = pd.read_csv(f, delim_whitespace=True, header=None, names=cols)
    method = this_df['Method'].unique().tolist()[0]
    if method in name_map:
        method = name_map[method]
    this_df = getEF(this_df, R)[[EFname, 'Target']]
    this_df.rename(columns={EFname: method}, inplace=True)
    max_dfs[method] = this_df

# read in min files
found_match = []
for f in args.min:
    this_df = pd.read_csv(f, delim_whitespace=True, header=None, names=cols)
    method = this_df['Method'].unique().tolist()[0]
    match_method = method.split('-min')[0].split('-then')[0]
    if match_method in name_map:
        match_method = name_map[match_method]
    # make sure we have paired data
    assert match_method in max_dfs, '%s missing from max data' %match_method
    this_df = getEF(this_df, R)[[EFname, 'Target']]
    this_df.rename(columns={EFname: 'min'}, inplace=True)
    # merge the pairs by target
    max_dfs[match_method] = pd.merge(max_dfs[match_method], this_df, on=['Target'], sort=False)
    found_match.append(match_method)

found_match = set(found_match)
assert found_match == set(max_dfs), 'Missing min data for some max data'

total_plots = len(found_match)
if found_match == set(paper_methods):
    grid_width = 4
    grid_length = 3
    found_match = paper_methods
else:
    grid_width = int(math.ceil(math.sqrt(total_plots)))
    grid_length = int(math.ceil(float(total_plots)/grid_width))

fig,ax = plt.subplots(figsize=(16,13))
# one plot per method
sub_axs = []
for plot_num,method in enumerate(found_match):
    sub_ax = plt.subplot2grid((grid_length,grid_width),
            (plot_num // grid_width, plot_num % grid_width),
            fig=fig)
    # sort descending by score from max
    sub_df = max_dfs[method].sort_values(by=method, inplace=False)
    sub_df['Target'] = sub_df['Target'].str.replace('_', ' ', regex=False)
    # target on y axis, EF on x
    targets = sub_df['Target'].tolist()
    y = list(range(len(targets)))

    # plot black dot at max, 
    sub_ax.plot(sub_df[method].tolist(), y, marker='o', color='black', ms=6,
            zorder=1, linestyle='None')
    
    # white dot at min, 
    sub_ax.plot(sub_df['min'].tolist(), y, marker='o', color='white', zorder=1,
    mec='black', mew=0.5, ms=6, linestyle='None')
    
    # thick method color line connecting them
    for i,target in enumerate(targets):
        if i == 0:
            label = method
        else:
            label = ''
        t_df = sub_df.loc[sub_df['Target'] == target]
        color = paper_palettes[method] if method in paper_palettes else \
                backup_palette[plot_num % len(backup_palette)]
        sub_ax.plot([t_df[method],t_df['min']], [i,i], color=color,
        linestyle='-', linewidth=4, alpha=0.7, zorder=3, label=label)
    sub_ax.set_xlim((-1,35))
    sub_axs.append((targets,sub_ax))

fig.legend(bbox_to_anchor=(1.05,0.35), ncol=3)
plt.tight_layout()
fig.canvas.draw()
for ynames,sub_ax in sub_axs:
    sub_ax.set_yticks(y)
    sub_ax.set_yticklabels(ynames)
fig.subplots_adjust(wspace=0.5)
fig.legend([black_circle, white_circle], ['Performance with top-ranked pose',
    'Performance with bottom-ranked pose'], frameon=True, 
    bbox_to_anchor=(0.95, 0.1)) 
fig.savefig('max_vs_min_pose_EF.pdf', bbox_inches='tight')
