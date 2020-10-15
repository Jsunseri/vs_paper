#!/bin/env python
import math,random,multiprocessing
from argparse import ArgumentParser
from collections import namedtuple
import pandas as pd
import numpy as np
from sklearn.utils import resample
from early_enrichment import getEF

OutStats = namedtuple('OutStats', ['low', 'high', 'p'])

def bootstrap(dfs, target, niters):
    boot_diffs = np.empty((niters))
    for i in range(niters):
        # bootstrap 95% CI of difference in test statistic:
        # draw N resamples with replacement from the original data; because we have two
        # classes do a stratified bootstrap respecting the original class distribution
        # (partly because otherwise there's a high risk of many samples having no
        # actives)
        boot_ef = pd.DataFrame()
        efs = []
        for method,df in dfs.items():
            this_df = df.loc[df['Target'] == target]
            n_samples = this_df.shape[0]
            preds,labels = resample(this_df['Prediction'], this_df['Label'], replace=True,
                    n_samples=n_samples, stratify=this_df['Label'])
            boot_df = pd.DataFrame({'Target': [target] * n_samples, 'Label':
                labels, 'Prediction': preds})
            this_ef = getEF(boot_df)
            efs.append(this_ef[EFname].tolist()[0])
        boot_diffs[i] = efs[0] - efs[1]
    return boot_diffs

def permutation(ranks, niters, subset_size, total_actives, R):
    assert len(ranks) % 2 == 0, 'ranks array has shape %d, which is unexpected' %len(ranks)
    n = len(ranks) // 2

    permute_out = np.empty((niters))
    #   repeat N_{B} times:
    for i in range(niters):
    #   - randomly permute the pooled data
        rng = np.random.default_rng()
        rng.shuffle(ranks)
    #   - compute the sample statistic for the first n1 observations from the pool
    #     and the second n2 observations and record the difference
        s1 = ranks[:n]
        top_actives = (s1 < subset_size).sum()
        ef1 = top_actives / (total_actives * R)
        s2 = ranks[n:]
        top_actives = (s2 < subset_size).sum()
        ef2 = top_actives / (total_actives * R)
        permute_out[i] = ef1 - ef2
    return permute_out

if __name__ == '__main__':
    parser = ArgumentParser(description='Use the permutation test to compute a '
            'p value for the difference between a pair of scoring methods, or '
            'use stratified bootstrap to compute a confidence interval on the '
            'difference between them.  Currently just supporting EF1% as the '
            'statistic used to compare them')
    parser.add_argument('-s', '--summary', nargs=2, help='Summary files for the '
            'two methods to be compared, arranged [LABEL PREDICTION TARGET TITLE '
            'METHOD]')
    parser.add_argument('-n', '--niters', type=int, default=1000, help='Number of '
            'bootstrap iterations to run')
    parser.add_argument('-nc', '--ncpus', type=int, default=1, help='Number of CPUs to '
            'run bootstrap sampling in parallel')
    parser.add_argument('--alpha', type=float, default=0.05, help='Alpha value '
            'for confidence intervals, default to 0.05 for 95% CI')
    parser.add_argument('-o', '--outprefix', default='', help='Output prefix '
            'for name of summary file')
    args = parser.parse_args()

    cols = ['Label', 'Prediction', 'Target', 'Title', 'Method']
    R = 0.01
    pctg = int(R * 100)
    EFname = 'EF{}%'.format(pctg) 
    dfs = {}
    methods = []
    for i,summary in enumerate(args.summary):
        df = pd.read_csv(summary, header=None, names=cols, delim_whitespace=True)
        df['Rank'] = df.groupby(['Target'])['Prediction'].rank(method='first', ascending=False)
        method = df['Method'].unique().tolist()[0]
        dfs[method] = df
        methods.append(method)

        this_ef = getEF(df)[[EFname, 'Target', 'NA', 'sizeR']]
        this_ef.rename(columns={EFname: method}, inplace=True)
        try:
            EFs = pd.merge(EFs, this_ef[[method, 'Target']], on=['Target'])
        except NameError:
            EFs = this_ef
    
    EFs['diff'] = EFs[methods[0]] - EFs[methods[1]]
    out_data = {}
    niters = args.niters
    targets = EFs['Target'].unique().tolist()
    pool = None
    if args.ncpus > 1:
        pool = multiprocessing.Pool(args.ncpus)
    for target in targets:
        test = EFs.loc[EFs['Target'] == target]['diff'].tolist()[0]
        if args.ncpus <= 1:
            result = bootstrap(dfs, target, niters)
        else:
            iter_chunk = int(np.ceil(niters * 1.0 / args.ncpus))
            result = []
            for i in range(args.ncpus):
                out = pool.apply_async(bootstrap, (dfs, target, iter_chunk))
                result.append(out)
            result = np.hstack([out.get() for out in result])
        # compute CI using the empirical method
        low = 2 * test - np.percentile(result, 100 * (1 - args.alpha / 2.))
        high = 2 * test - np.percentile(result, 100 * (args.alpha / 2.))
        if low > high:
            low,high = high,low
    
        if test == 0:
            # the test will always pass
            p = 1
        else:
            # permutation test using the ranks of actives:
            # - pool the data (i.e. the active ranks)
            this_ef = EFs.loc[EFs['Target'] == target]
            subset_size = this_ef['sizeR'].values[0]
            total_actives = this_ef['NA'].values[0]
            ranks = np.empty((total_actives*2))
            for i,(method,df) in enumerate(dfs.items()):
                this_df = df.loc[df['Target'] == target]
                start = i * total_actives
                end = i * total_actives + total_actives
                ranks[start:end] = this_df.loc[this_df['Label'] == 1]['Rank'].to_numpy()
            if args.ncpus <= 1:
                result = permutation(ranks, niters, subset_size,
                        total_actives, R)
            else:
                iter_chunk = int(np.ceil(niters * 1.0 / args.ncpus))
                result = []
                for i in range(args.ncpus):
                    out = pool.apply_async(permutation, (ranks, iter_chunk,
                        subset_size, total_actives, R))
                    result.append(out)
                result = np.hstack([out.get() for out in result])

                # - the p-value is the observed frequency of differences where
                #   diff_{permutation} >= diff_{test}
                p = (
                    len(result[np.where(result >= math.fabs(test))]) +
                    len(result[np.where(result <= -math.fabs(test))])
                ) / len(result)
        out_data[target] = OutStats(low,high,p)
  
    if pool:
        pool.close()
    with open('%spvalue_and_ci.csv' %args.outprefix, 'w') as f:
        for target,data in out_data.items():
            f.write('%s %0.3f %0.3f %0.3f %0.3f\n' %(target,test,data.low, data.high, data.p))
