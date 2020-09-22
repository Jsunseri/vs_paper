#!/usr/bin/env python
import matplotlib as mpl
mpl.use('Agg')
import os,sys,math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from vspaper_settings import paper_palettes, name_map

# you pass in the auc_bytarget files, we make a stripplot by target
df = pd.DataFrame()
for csv in sys.argv[1:]:
    method = csv.split('_bytarget')[0]
    try:
        method = name_map[method]
    except KeyError:
        pass
    names = ['Target', 'AUC', 'APS']
    tmp = pd.read_csv(csv, delim_whitespace=True, header=None, names=names)
    tmp['Method'] = method
    df = pd.concat([df, tmp], ignore_index=True)

df['Target'] = df['Target'].replace('_', ' ', regex=True)
# now sort targets by increasing median AUC
grouped = df.groupby(['Target'], as_index=False)
medians = grouped['AUC'].median()
medians.sort_values(by='AUC', inplace=True)
sorter = medians['Target'].tolist()
sorter_index = dict(zip(sorter,range(len(sorter))))
df['tmp_rank'] = df['Target'].map(sorter_index)
df.sort_values(['tmp_rank'], ascending=True, inplace=True)
df.drop('tmp_rank', 1, inplace=True)

fig,ax = plt.subplots(figsize=(16, 16))
sns.stripplot(x="AUC", y="Target", hue="Method", data=df,
        palette=paper_palettes, alpha=0.7, size=10, ax=ax)
ax_ylims = ax.get_ylim()
ax.plot([0.5, 0.5], [ax_ylims[0], ax_ylims[1]], linestyle='--', color='gray', lw=3,
                    zorder=1, alpha=0.5)
ax.set_ylim(ax_ylims)
ax.legend(title='Method', frameon=True, loc='lower left', ncol=3)
fig.savefig('AUC_stripplot.pdf', bbox_inches='tight')
