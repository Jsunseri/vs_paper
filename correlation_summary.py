#!/bin/env python
import sys
import pandas as pd
from packaging import version

# requires pandas>=1.1.0 because of list arg to pandas::DataFrame::pivot index
assert version.parse(pd.__version__) >= version.parse('1.1.0'), 'pandas version >= 1.1.0 required'

# you provide the summary files, they need to be 
# LABELS PREDICTIONS TARGET TITLE METHOD
cols = ['Label', 'Prediction', 'Target', 'Title', 'Method']
summary_files = sys.argv[1:]
df = pd.DataFrame()
for fname in summary_files:
    this_df = pd.read_csv(fname, delim_whitespace=True, header=None, names=cols)
    this_df.drop_duplicates(subset=['Label', 'Target', 'Title', 'Method'], inplace=True, ignore_index=True)
    df = pd.concat([df, this_df], ignore_index=True)

try:
    df = df.pivot(index=['Label', 'Target', 'Title'], columns='Method', values='Prediction')
except Exception as e:
    print(e)
    print('Checking for duplicates...')
    df = df.set_index(['Label', 'Target', 'Title'])
    print(df[df.index.duplicated()].to_string())
    sys.exit()

# print out some of the dataframe as a sanity check
print(df.head())
print(df.tail())

# full correlation
print('Pearson correlation:\n')
print(df.corr(method='pearson').to_string())

print('Spearman correlation:\n')
print(df.corr(method='spearman').to_string())

print('Kendall Tau correlation:\n')
print(df.corr(method='kendall').to_string())

# actives only
print('All actives, Pearson correlation:\n')
print(df.loc[df['Label']==1].corr(method='pearson').to_string())

print('All actives, Spearman correlation:\n')
print(df.loc[df['Label']==1].corr(method='spearman').to_string())

print('All actives, Kendall Tau correlation:\n')
print(df.loc[df['Label']==1].corr(method='kendall').to_string())

# inactives only
print('All inactives, Pearson correlation:\n')
print(df.loc[df['Label']==0].corr(method='pearson').to_string())

print('All inactives, Spearman correlation:\n')
print(df.loc[df['Label']==0].corr(method='spearman').to_string())

print('All inactives, Kendall Tau correlation:\n')
print(df.loc[df['Label']==0].corr(method='kendall').to_string())

# maybe just actives/inactives for targets a given method does well on?

# intersection/union for top 1% of actives per target
# print('Intersection over union for top 1% of actives per target:\n')
# nonm = ['Label', 'Prediction', 'Target', 'Title']
# mnames = [item for item in df.columns if item not in nonm]
# print('%s\n' %(' '.join(mnames)))
# for mn in mnames:
#     grouped = df['Method'].groupby(['Target'])
#     topn = grouped.apply(lambda x: x.nlargest(int(len(x) * .01), 'Prediction'))
#     for mn in mnames:
#         grouped = df.groupby(['Target'])
#         topn = grouped.apply(lambda x: x.nlargest(int(len(x) * .01), 'Predictions'))

# intersection/union for bottom 1% of inactives per target
# print('Intersection over union for bottom 1% of inactives per target:\n')
