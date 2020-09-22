import pickle
import numpy as np
import sys
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from vspaper_settings import paper_palettes, name_map

scorecolors = {'inactive': sns.color_palette()[3], 'active': sns.color_palette()[2]}

def make_plot(df, cols, figname):
    ncols = len(cols)
    cdict = scorecolors
    g = sns.pairplot(df, vars=cols, hue='label', diag_kind="kde",
            palette={0: cdict['inactive'], 1: cdict['active']},
            plot_kws=dict(edgecolor="none", alpha=0.7, marker='.'), 
            diag_kws=dict(shade=True))
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    for i in range(0, ncols):
        for j in range(0, ncols):
            if i != j:
                r, _ = pearsonr(df[cols[i]].values, df[cols[j]].values)
                g.axes[i,j].annotate(r'$\rho = {0:.2}$'.format(r), xy=(.1, .9),
                        xycoords=g.axes[i,j].transAxes, bbox=props)
            g.axes[i,j].set(ylim=(df[cols[i]].min(), df[cols[i]].max()))
            g.axes[i,j].set(xlim=(df[cols[j]].min(), df[cols[j]].max()))
    g.savefig(figname, dpi=300, bbox_inches='tight')

# load preds df
df = pickle.load(open('preds_df.cpickle', 'rb'))
newmethods = ["dense", "crossdock_default2018", "general_default2018"]
cols = list(df)
shared = ["Title", "Target", "label"]
methods = []
variances = []
for elem in shared:
    assert elem in cols, "Compound title and target name must be columns in the DataFrame to proceed"

# compute means, this is annoying, precompute this in pickle in future
for d in newmethods:
    for stype in ['CNNscore', 'CNNaffinity']:
        snames = ['%s_seed%d_%s' %(d, seed, stype) for seed in list(range(5))]
        mname = name_map[d + '-' + stype + '-mean']
        df[mname] = df[snames].mean(axis=1)
        methods.append(mname)

# make df of preds per target/compound pair
# make df of gap per target/compound pair
for method in methods:
    # make tmp dataframe for each method
    tmpdf = df[shared + [method]]
    # groupby Title and Target,
    # compute diff between score between top and bottom rank
    thisgap_df = tmpdf.groupby(shared)[method].agg(np.ptp).reset_index()
    # merge into new dataframe which is Target Title label Diff Method 
    try:
        gap_df = pd.merge(gap_df, thisgap_df[shared + [method]], on=shared)
    except NameError:
        gap_df = thisgap_df[shared + [method]]

    # now make df of preds
    thispred_df = tmpdf.loc[tmpdf.groupby(shared, as_index=False)[method].idxmax()]
    thispred_df = thispred_df.melt(id_vars=shared, var_name="Method",
            value_vars=[method], value_name="Prediction")
    try:
        # pred_df = pd.merge(pred_df, thispred_df, on=shared)
        pred_df = pd.concat([pred_df, thispred_df], ignore_index=True,
                sort=False)
    except NameError:
        pred_df = thispred_df

# make_plot(gap_df, methods, 'pose_scoregap_correlation.png')

# gap between max and min score, as a function of max score
# need aggregate df that has Target Title label Diff Max Method
scorediff_pct = r'$Score_{Max} - Score_{Min}$'
gap_df = gap_df.melt(id_vars=shared, var_name="Method", value_vars=methods,
        value_name=scorediff_pct)

final_df = pred_df.merge(gap_df, on=shared + ["Method"])
# final_df[[scorediff_pct]] = final_df[[scorediff_pct]].div(final_df[['Prediction']].values, axis=0)

# I want to plot this probably as a grid of kdeplots? don't really care about
# the correlation here 
# hardcoding for now, TODO: fix?
grid_length = 3
grid_width = 2 # for now
fig,ax = plt.subplots(figsize=(16,16))
props = dict(boxstyle='round', facecolor='white', alpha=0.8)
sorted_methods = [
        'Dense\n(Pose)', 'Dense\n(Affinity)', 
        'Cross-Docked\n(Pose)', 'Cross-Docked\n(Affinity)', 
        'General\n(Pose)', 'General\n(Affinity)'
        ]
for i,method in enumerate(sorted_methods):
    plot_num = i
    sub_ax = plt.subplot2grid((grid_length,grid_width),
            (plot_num // grid_width, plot_num % grid_width),
                            fig=fig)
    subframe = final_df.loc[final_df.Method == method]
    sns.kdeplot(subframe['Prediction'], subframe[scorediff_pct], 
            ax = sub_ax,
            shade=True,
            shade_lowest=False, 
            label = method, 
            alpha=0.7,
            color=paper_palettes[method])
    # sns.scatterplot(subframe['Prediction'], subframe[scorediff_pct], 
            # ax = sub_ax,
            # hue = method, 
            # alpha=0.7,
            # linewidth=0,
            # palette=paper_palettes
            # )
    r, _ = pearsonr(subframe['Prediction'].values, subframe[scorediff_pct].values)
    sub_ax.annotate(r'$\rho = {0:.2}$'.format(r), xy=(.1, .9),
             xycoords=sub_ax.transAxes, bbox=props)
    if i == 0:
        sub_ax.set_title('Pose')
    if i == 1:
        sub_ax.set_title('Affinity')

fig.savefig('pose_scoregap_vs_pred.pdf', bbox_inches='tight')
