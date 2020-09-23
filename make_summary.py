#! /usr/bin/env python
import sys,os,pickle
import numpy as np
import pandas as pd

def output_summary(df, grouped, d):
    ov = df.loc[grouped[d].idxmax()]
    ov['method'] = d
    ov.to_csv(path_or_buf='%s-then-max.summary'%(d), sep=' ', header=False,index=False,
            columns=['label', d, 'Target', 'Title', 'method'])
    # if we're doing CNNaffinity, also use CNNscore to rank
    altmethod = d.replace('affinity', 'score')
    if altmethod != d:
        ov = df.loc[grouped[altmethod].idxmax()]
        ov['method'] = d
        ov.to_csv(path_or_buf='%s-then-max_%s.summary'%(d,"scorerank"), sep=' ', header=False,index=False,
                columns=['label', d, 'Target', 'Title', 'method'])
    return ov

df = pickle.load(open('preds_df_withmissing.cpickle', 'rb'))
newmethods = ["dense", "crossdock_default2018", "general_default2018"]
cols = list(df)

# let's look at mean across seeds and then max within groupby as well as
# max per seed within groupby and then mean across seeds
for d in newmethods:
    for stype in ['CNNscore', 'CNNaffinity']:
        snames = ['%s_seed%d_%s' %(d, seed, stype) for seed in list(range(5))]
        df[d + '-' + stype + '-mean'] = df[snames].mean(axis=1)

grouped = df.groupby(['Target','Title'], as_index=False)
ov = df.loc[grouped['Vina'].idxmax()]
ov['method'] = 'Vina'
ov.to_csv(path_or_buf='vina.summary', sep=' ', header=False, index=False,
        columns=['label', 'Vina', 'Target', 'Title', 'method'])

for d in newmethods:
    for stype in ['CNNscore', 'CNNaffinity']:
        # loc of max along the mean val
        output_summary(df, grouped, '%s-%s-mean' %(d, stype))
