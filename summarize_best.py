#!/bin/env python
import sys
import pandas as pd

# this is for the "simple descriptor" and fingerprint models - to use them as a
# baseline, i want to just show the max performance per target over each
# descriptor type
maxcols = []
for f in sys.argv[1:]:
    with open(f, 'r') as tmpf:
        for line in tmpf:
            contents = line.split()
            ncols = len(contents)
            break
    method = f.split('EF1')[0]
    cols = ['Target', f, '%s-NEF' %method]
    maxcols.append(f)
    df = pd.read_csv(f, delim_whitespace=True, names=cols[:ncols], header=None)
    try:
        alldf = pd.merge(alldf, df, on='Target', sort=False)
    except NameError:
        alldf = df

alldf['max'] = alldf[maxcols].max(axis=1)
alldf['max-method'] = alldf[maxcols].idxmax(axis=1)
alldf.to_csv('best_per_target.csv', sep=' ', header=False, index=False,
        columns=['Target', 'max', 'max-method'])
