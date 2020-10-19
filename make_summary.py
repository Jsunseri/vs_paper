#! /usr/bin/env python
import sys,os,pickle
from argparse import ArgumentParser
import numpy as np
import pandas as pd

def output_max_summary(df, grouped, d):
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

def output_min_summary(df, grouped, d):
    ov = df.loc[grouped[d].idxmin()]
    ov['method'] = d
    ov.to_csv(path_or_buf='%s-then-min.summary'%(d), sep=' ', header=False,index=False,
            columns=['label', d, 'Target', 'Title', 'method'])
    # if we're doing CNNaffinity, also use CNNscore to rank
    altmethod = d.replace('affinity', 'score')
    if altmethod != d:
        ov = df.loc[grouped[altmethod].idxmin()]
        ov['method'] = d
        ov.to_csv(path_or_buf='%s-then-min_%s.summary'%(d,"scorerank"), sep=' ', header=False,index=False,
                columns=['label', d, 'Target', 'Title', 'method'])
    return ov

if __name__ == '__main__':
    parser = ArgumentParser(description='Take pickle of full results and output '
            'summary files per method')
    parser.add_argument('-p', '--pickle', type=str, help='Pickle '
            'with data to summarize')
    parser.add_argument('-c', '--csv', nargs='*', help='CSVs with data to '
            'summarize; if provided, these should all be associated with the '
            'same method')
    parser.add_argument('-j', '--justseeds', action='store_true', help='Do not '
            'calculate CNN ensemble values, just dump summary files for the '
            'individual seeds')
    args = parser.parse_args()
    exclude = ['Rank', 'Title', 'MW', 'Target', 'File']
    methods = ['Vina', 'RFScore-4', 'RFScore-VS']
    stypes = ['CNNscore', 'CNNaffinity']

    assert args.pickle or args.csv, 'No input data provided'
    if args.pickle:
        df = pickle.load(open(args.pickle, 'rb'))
        cols = list(df)
        # assume ensemble columns (those with 'seed' in the name) should be
        # used together to generate predictions
        cnns = set([name.split('_seed')[0] for name in cols if 'CNNscore' in name])
        if not args.justseeds:
            for d in cnns:
                for stype in ['CNNscore', 'CNNaffinity']:
                    snames = ['%s_seed%d_%s' %(d, seed, stype) for seed in list(range(5))]
                    print('Computing ensemble mean for %s-%s' %(d,stype))
                    df[d + '-' + stype + '-mean'] = df[snames].mean(axis=1)

                    print('Computing ensemble median for %s-%s' %(d,stype))
                    df[d + '-' + stype + '-median'] = df[snames].median(axis=1)

                    print('Computing ensemble max for %s-%s' %(d,stype))
                    df[d + '-' + stype + '-max'] = df[snames].max(axis=1)

                    print('Computing ensemble min for %s-%s' %(d,stype))
                    df[d + '-' + stype + '-min'] = df[snames].min(axis=1)
    
        grouped = df.groupby(['Target','Title'], as_index=False)
        if not args.justseeds:
            for method in methods:
                print('Writing out summary max file for %s' %method)
                ov = df.loc[grouped[method].idxmax()]
                ov['method'] = method
                ov.to_csv(path_or_buf='%s-max.summary' %method, sep=' ', header=False, index=False,
                        columns=['label', method, 'Target', 'Title', 'method'])

                print('Writing out summary min file for %s' %method)
                ov = df.loc[grouped[method].idxmin()]
                ov['method'] = method
                ov.to_csv(path_or_buf='%s-min.summary' %method, sep=' ', header=False, index=False,
                        columns=['label', method, 'Target', 'Title', 'method'])
        
        for d in cnns:
            for stype in ['CNNscore', 'CNNaffinity']:
                if not args.justseeds:
                    for stat in ['mean', 'median', 'max', 'min']:
                        print('Writing out summary max file for %s-%s-%s' %(d,stype,stat))
                        output_max_summary(df, grouped, '%s-%s-%s' %(d, stype, stat))
                        print('Writing out summary min file for %s-%s-%s' %(d,stype,stat))
                        output_min_summary(df, grouped, '%s-%s-%s' %(d, stype, stat))
                else:
                    snames = ['%s_seed%d_%s' %(d, seed, stype) for seed in list(range(5))]
                    for seed in snames:
                        print('Writing out summary max file for %s' %(seed))
                        output_max_summary(df, grouped, '%s' %(seed))
                        print('Writing out summary min file for %s' %(seed))
                        output_min_summary(df, grouped, '%s' %(seed))
    if args.csv:
        df = pd.DataFrame()
        for csv in args.csv:
            df = pd.concat([df, pd.read_csv(csv, 
                delim_whitespace=True, header=0)], ignore_index=True)
        
        cols = list(df)
        if 'Vinardo' in cols:
            df.loc[:,'Vinardo'] = df[['Vinardo']].mul(-1) 
        df["label"] = [0 if (("decoy" in item) or ('inactive' in item)) else 1 for item in df["File"]]
        methods = set(cols) - set(exclude)

        for method in methods:
            grouped = df.groupby(['Target','Title'], as_index=False)
            ov = df.loc[grouped[method].idxmax()]
            ov['method'] = method
            print('Writing out summary max file for %s' %method)
            ov.to_csv(path_or_buf='%s-max.summary' %method, sep=' ', header=False, index=False,
                    columns=['label', method, 'Target', 'Title', 'method'])

            ov = df.loc[grouped[method].idxmin()]
            ov['method'] = method
            print('Writing out summary min file for %s' %method)
            ov.to_csv(path_or_buf='%s-min.summary' %method, sep=' ', header=False, index=False,
                    columns=['label', method, 'Target', 'Title', 'method'])
