#!/bin/env python
from argparse import ArgumentParser
import pandas as pd
import numpy as np
from sklearn.utils import resample
from early_enrichment import getEF

if __name__ == '__main__':
    parser = ArgumentParser(description='Use the permutation test to compute a '
            'p value for the difference between a pair of scoring methods, or '
            'use stratified bootstrap to compute a confidence interval on the '
            'difference between them.  Currently just supporting EF1% as the '
            'statistic used to compare them')
    parser.add_argument('-s', '--summary', nargs=2, help='Summary files for the '
            'two methods to be compared, arranged [LABEL PREDICTION TARGET TITLE '
            'METHOD]')
    parser.add_argument('--alpha', type=float, default=0.05, help='Alpha value '
            'for confidence intervals, default to 0.05 for 95% CI')
    parser.add_argument('-o', '--outprefix', default='', help='Output prefix '
            'for name of summary file')

    OutStats = namedtuple('OutStats', ['low', 'high', 'p'])
    cols = ['Label', 'Prediction', 'Target', 'Title', 'Method']
    R = 0.01
    pctg = int(R * 100)
    EFname = 'EF{}%'.format(pctg) 
    dfs = {}
    methods = []
    for i,summary in enumerate(args.summary):
        df = pd.read_csv(summary, header=None, names=cols, delim_whitespace=True)
        method = df['Method'].unique().tolist()[0]
        dfs[method] = df
        methods.append(method)
        this_ef = getEF(df)[[EFname, Target]]
        this_ef.rename(columns={EFname: method}, inplace=True)
        try:
            EFs = pd.merge(EFs, getEF(df), on=['Target'])
        except NameError:
            EFs = getEF(df)
    
    EFs['diff'] = EFs[methods[0]] - EFs[methods[1]]
    boot_diffs = []
    out_data = {}
    niters = 1000 
    # TODO: could parallelize over targets, we'll see how slow it all is
    for target in targets:
        for i in range(niters):
            boot_ef = None
            # bootstrap 95% CI of difference in test statistic:
            # draw N resamples with replacement from the original data; because we have two
            # classes do a stratified bootstrap respecting the original class distribution
            # (partly because otherwise there's a high risk of many samples having no
            # actives)
            for method,df in dfs.items():
                this_df = df.loc[df['Target'] == target]
                n_samples = this_df.shape[0]
                preds,labels = resample(this_df['Prediction'], this_df['Label'], replace=True,
                        n_samples=n_samples, stratify=this_df['Label'])
                boot_df = pd.DataFrame({'Target': [target] * n_samples, 'Label':
                    labels, 'Prediction': preds})
                this_ef = getEF(boot_df)[[EFname, Target]]
                this_ef.rename(columns={EFname: method}, inplace=True)
                try:
                    boot_ef = pd.merge(boot_ef, this_ef, on=['Target'])
                except NameError:
                    boot_ef = this_ef
            boot_diffs.append((boot_ef[methods[0]] - boot_ef[methods[1]]).tolist()[0])
        test = EFs.loc[EFs['Target'] == target]['diff'].tolist()[0]
        # compute CI either from 2.5% and 97.5% percentile values (not recommended)
        # or using the empirical method
        low = 2 * test - np.percentile(boot_diffs, 100 * (1 - alpha / 2.))
        high = 2 * test - np.percentile(boot_diffs, 100 * (alpha / 2.))
        if low > high:
            low,high = high,low
        out_data[target] = OutStats(low,high,np.nan)
    
        # permutation test using the ranks of actives:
        # - compute difference in test statistics
        # - pool the data (i.e. the active ranks)
        #   repeat N_{B} times:
        #   - randomly permute the pooled data
        #   - compute the sample statistic for the first n1 observations from the pool
        #     and the second n2 observations and record the difference
        # - the p-value is the observed frequency of differences where
        #   diff_{permutation} >= diff_{test}
    
    with open('%spvalue_and_ci.csv' %outprefix, 'w') as f:
        for target,data in out_data:
            f.write('%s %0.3f %0.3f %0.3f\n' %(target,data.low, data.high, data.p))
