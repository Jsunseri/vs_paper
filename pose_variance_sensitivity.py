#! /usr/bin/env python
import sys,pickle
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score
from vspaper_settings import paper_palettes, name_map, reverse_map, swarm_markers, litpcba_order

# this time let's look at both AUC and EF1% when we drop percentiles based on
# variance, and let's also look at mean variance in each quartile

def calc_auc(target_predictions):
    # print target_and_method
    y_true=[]
    y_score=[]
    for i,item in enumerate(target_predictions):
        try:
            label = float(item[0])
            score = float(item[1])
            y_true.append(label)
            y_score.append(score)
        except Exception as e:
            print('Error: %d %f %s\n'%(label, score))
            continue
    fpr,tpr,_ = roc_curve(y_true,y_score)
    return roc_auc_score(y_true,y_score)

newmethods = ["dense", "crossdock_default2018", "general_default2018"]
ensembles = {}
for n in newmethods:
    for stype in ['CNNscore', 'CNNaffinity']:
        ename = n + "\_%s\_variance" %stype
        ensembles[ename] = []
        for seed in range(5):
            colname = "%s%d_%s" %(n,seed,stype)
            cols.append(colname)
            ensembles[ename].append(colname)
df = pd.read_csv("sdsorter_ds.summary", delim_whitespace=True, header=None,
        names=cols)
for varname,elist in ensembles.items():
    df[varname] = df[elist].var(axis=1)

df = pickle.load(open('preds_df.cpickle', 'rb'))
# find quartiles of variance and remove the top 1st, 2nd, and 3rd before
# making predictions
# i've been taking the max over the seeds and then computing the mean
# here let's do both
ptile_data = []
for ptile in [.25, .5, .75, 1]:
    print('calculating percentile %f'%ptile)
    for vname,elist in ensembles.items():
        cols = ['target', 'compound', 'label'] + elist
        method = vname.replace('variance', 'mean')
        meanmax = df.groupby(['target', 'compound', 'label']).apply(lambda x : \
                          x[x[vname]<x[vname].quantile(ptile)][elist].mean(axis=1).max())
        meanmax = meanmax.reset_index()
        meanmax = meanmax.rename(columns={0:method})
        meanmax.fillna(0, inplace=True)
        maxmean = df.groupby(['target', 'compound', 'label']).apply(lambda x : \
                x[x[vname]<x[vname].quantile(ptile)][elist].max()).mean(axis=1)
        maxmean = maxmean.reset_index()
        maxmean = maxmean.rename(columns={0:method})
        maxmean.fillna(0, inplace=True)
        targets = maxmean['target'].unique()
        for target in targets:
            auc = calc_auc(zip(maxmean[maxmean['target'] == target]['label'].values, maxmean[maxmean['target'] == target][method].values))
            ptile_data.append((True, method, target, ptile, auc))
            auc = calc_auc(zip(meanmax[meanmax['target'] == target]['label'].values, meanmax[meanmax['target'] == target][method].values))
            ptile_data.append((False, method, target, ptile, auc))

pickle.dump(ptile_data, open('percentile_aucs.pickle', 'wb'), -1)
fig,ax = plt.subplots()
summarydf = pd.DataFrame.from_records(ptile_data, columns=['maxfirst', 'method', 'target',
    'variance\_percentile', 'AUC'])
for first in [True, False]:
    plt.cla()
    sns.violinplot(x="method", y="AUC", hue="variance\_percentile",
                          data=summarydf[summarydf['maxfirst'] == first],
                          ax=ax)
    labelname = 'maxfirst' if first else 'meanfirst'
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)
    fig.savefig('variance_percentile_%s.pdf' %labelname, bbox_inches='tight')
