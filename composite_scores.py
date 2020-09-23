#! /usr/bin/env python
from argparse import ArgumentParser
import pickle

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from vspaper_settings import paper_palettes, name_map, reverse_map, swarm_markers, litpcba_order

parser = ArgumentParser(description='Generate predictions using some composite score methods')
parser.add_argument('-p', '--pickle', type=str, required=True, help='Pandas '
        'DataFrame pickle of results for all poses')
args = parser.parse_args()

df = pickle.load(open(args.pickle, 'rb'))
cnns = ["dense", "crossdock_default2018", "general_default2018"]

# our "composite" predictions will be:
#
# (1) product of score and affinity across poses, then max within groupby as pred
#
# (2) score_mean.div(score_std), taking the max wrt this recalibrated score directly
#
# (3) max within groupby for score and affinity separately, then product for prediction
#
# (4) score-max * (score-max - score-min) [i.e. the score-gap]
#
# (5) score_mean.div(score_std), taking the max wrt the mean and then using this 
#     recalibrated score as the pred
outpreds = []
for d in cnns:
    means = []
    for stype in ['CNNscore', 'CNNaffinity']:
        print('Computing mean and stdev for %s-%s' %(d,stype))
        snames = ['%s_seed%d_%s' %(d, seed, stype) for seed in list(range(5))]
        mname = d + '-' + stype + '-mean'
        stdname = d + '-' + stype + '-std'
        recalibrated_score = d + '-' + stype + '-mean-div-std'
        df[mname] = df[snames].mean(axis=1)
        means.append(mname)
        df[stdname] = df[snames].std(axis=1)
        df[recalibrated_score] = df[mname] / df[stdname]
        outpreds.append(recalibrated_score)
    poselevel_product = d + '-CNNscore_CNNaffinity-poselevel_product'
    df[poselevel_product] = df[means[0]] * df[means[1]]
    outpreds.append(poselevel_product)

grouped = df.groupby(['Target','Title'], as_index=False)
# take care of (1) and (2)
for out in outpreds:
    print('Writing out %s' %out)
    outdf = df.loc[grouped[out].idxmax()]
    outdf['Method'] = out
    outdf.to_csv(path_or_buf='%s.summary' %out, sep=' ', header=False,
            index=False, columns=['label', out, 'Target', 'Title', 'Method'])

# now (5), (4), and (3)
for d in cnns:
    for stype in ['CNNscore', 'CNNaffinity']:
        mname = d + '-' + stype + '-mean'
        scoregap = d + '-' + stype + '-scoregap'
        meanmax = d + '-' + stype + '-mean_max'
        meanmin = d + '-' + stype + '-mean_min'
        outdf = df.loc[grouped[mname].idxmax()]
        outdf = outdf.merge(df.loc[grouped[mname].idxmin()], on=['label',
            'Target', 'Title'], sort=False, suffixes=('_max', '_min'))
        # (5)
        recalibrated_score = d + '-' + stype + '-mean-div-std'
        outname = d + '-' + stype + 'maxthen-mean-div-std'
        outdf['Method'] = outname
        print('Writing out %s' %outname)
        outdf.to_csv(path_or_buf='%s.summary' %outname, sep=' ', header=False,
                index=False, columns=['label', recalibrated_score, 'Target', 'Title', 'Method'])
        # (4)
        outdf[scoregap] = outdf[meanmax] * (outdf[meanmax] - outdf[meanmin])
        outdf['Method'] = scoregap
        print('Writing out %s' %scoregap)
        outdf.to_csv(path_or_buf='%s.summary' %scoregap, sep=' ', header=False,
                index=False, columns=['label', scoregap, 'Target', 'Title', 'Method'])
    # (3)
    scoremean = d + '-CNNscore-mean'
    affmean = d + '-CNNaffinity-mean'
    outdf = df.loc[grouped[scoremean].idxmax()][['label', scoremean, 'Target', 'Title']]
    outdf = outdf.merge(df.loc[grouped[affmean].idxmax()], 
            on=['label', 'Target', 'Title'], sort=False)
    predlevel_product = d + '-CNNscore_CNNaffinity-predlevel_product'
    outdf[predlevel_product] = outdf[scoremean] * outdf[affmean]
    outdf['Method'] = predlevel_product
    print('Writing out %s' %predlevel_product)
    outdf.to_csv(path_or_buf='%s.summary' %predlevel_product, sep=' ', header=False,
            index=False, columns=['label', predlevel_product, 'Target', 'Title', 'Method'])
