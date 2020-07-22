import pickle
import numpy as np
import sys
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

plt.style.use('seaborn-white')
SMALL_SIZE=10
MEDIUM_SIZE=12
BIGGER_SIZE=12
SUBFIG_SIZE=12

plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

mpl.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
mpl.rc('text', usetex=True)
sns.set_palette(sns.color_palette("muted"))

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
paper_palettes['dense-CNNaffinity-mean'] = '#28B463'
paper_palettes['dense-aff-mean'] = '#28B463'
paper_palettes['dense'] = '#28B463'
paper_palettes['dense_consensus'] = '#cdf2dd'
paper_palettes['dense consensus'] = '#cdf2dd'
paper_palettes['crossdock_default2018-CNNscore-mean'] = '#E59866'
paper_palettes['crossdock_default2018-CNNaffinity-mean'] = '#BA4A00'
paper_palettes['crossdock default2018-CNNaffinity-mean'] = '#BA4A00'
paper_palettes['crossdocked2018-CNNaffinity-mean'] = '#BA4A00'
paper_palettes['crossdocked'] = '#BA4A00'
paper_palettes['crossdock default2018 consensus'] = '#f0c4a7'
paper_palettes['crossdock_default2018_consensus'] = '#f0c4a7'
paper_palettes['general_default2018-CNNscore-mean'] = '#D7BDE2'
paper_palettes['general_default2018-CNNaffinity-mean'] = '#9B59B6'
paper_palettes['general'] = '#9B59B6'
paper_palettes['general default2018-CNNaffinity-mean'] = '#9B59B6'
paper_palettes['generalset2018-CNNaffinity-mean'] = '#9B59B6'
paper_palettes['general_default2018_consensus'] = '#e1d2e9'
paper_palettes['general default2018 consensus'] = '#e1d2e9'
paper_palettes['rf-score-vs'] = '#D98880'
paper_palettes['rf-score-4'] = '#A93226'
paper_palettes['Dense (Pose)'] = '#82E0AA'
paper_palettes['Dense (Affinity)'] = '#28B463'
paper_palettes['Cross-Docked Set (Pose)'] = '#E59866'
paper_palettes['Cross-Docked Set (Affinity)'] = '#BA4A00'
paper_palettes['General Set (Pose)'] = '#D7BDE2'
paper_palettes['General Set (Affinity)'] = '#9B59B6'
paper_palettes['RFScore-VS'] = '#5DADE2'
paper_palettes['RFScore-4'] = '#2874A6'
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

for method in methods:
    # make tmp dataframe for each method
    tmpdf = df[shared + [method]]
    # sort by increasing score
    sortdf = tmpdf.sort_values(method, ascending=True)
    # groupby Title and Target,
    # compute diff between score between top and bottom rank
    newdf = sortdf.groupby(shared)[method].agg(np.ptp).reset_index()
    # newdf['Method'] = method
    # newdf.rename(columns={method: 'Diff'}, inplace=True)
    # concat to new dataframe which is Target Title label Diff Method 
    try:
        final_df = pd.merge(final_df, newdf[shared + [method]], on=shared)
    except NameError:
        final_df = newdf[shared + [method]]

make_plot(final_df, methods, 'pose_scoregap_correlation.png')
