#!/usr/bin/env python
import math
from argparse import ArgumentParser
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
from tabulate import tabulate
from early_enrichment import getEF
from vspaper_settings import paper_palettes, backup_palette, name_map, reverse_map

def check_cols(fname):
    cols = ['Label', 'Prediction', 'Target', 'Title', 'Method']
    altcols = ['Label', 'Prediction', 'Target', 'Method']
    # some of the outputs don't have moltitles, so check real quick
    with open(fname, 'r') as tmpf:
        for line in tmpf:
            contents = line.split()
            if len(contents) == len(cols):
                usecols = cols
            elif len(contents) == len(altcols):
                usecols = altcols
            else:
                sys.exit('Unknown columns in input file %s' %fname)
    return usecols

if __name__ == '__main__':
    parser = ArgumentParser(description='Grid of Per-target barplots (one plot '
             'per method) showing raw EF and EF difference compared with a provided '
             'reference')
    parser.add_argument('-s', '--summary', nargs='+', help='Summary files for '
            'all but the reference method ')
    parser.add_argument('-r', '--ref', required=True, help='Summary file for '
            'reference method')
    parser.add_argument('-o', '--outprefix', default='', help='Output prefix '
            'for plot name')
    args = parser.parse_args()

    cols = ['Label', 'Prediction', 'Target', 'Title', 'Method']
    altcols = ['Label', 'Prediction', 'Target', 'Method']
    R = 0.01
    pctg = int(R * 100)
    EFname = 'EF{}%'.format(pctg) 
    diffname = EFname

    black_square = mlines.Line2D([], [], color='black', marker='s', linestyle='None',
                                      markersize=16, label='Black squares')
    gray_square = mlines.Line2D([], [], color='gray', marker='s', linestyle='None',
                                      markersize=16, label='Gray squares')

    # read in ref
    usecols = check_cols(args.ref)
    df = pd.read_csv(args.ref, header=None, names=usecols, delim_whitespace=True)
    refmethod = df['Method'].unique().tolist()[0]
    if refmethod in name_map:
        refmethod = name_map[refmethod]
    ref_EF = getEF(df)[[EFname, 'Target']]
    ref_EF.rename(columns={EFname: refmethod}, inplace=True)

    total_plots = len(args.summary)
    grid_width = int(math.ceil(math.sqrt(total_plots)))
    grid_length = int(math.ceil(float(total_plots)/grid_width))
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    # fig,ax = plt.subplots(figsize=(16,12)) # general
    fig,ax = plt.subplots(figsize=(16,16)) # refined, crossdock
    for i,summary in enumerate(args.summary):
        usecols = check_cols(summary)
        df = pd.read_csv(summary, header=None, names=usecols, delim_whitespace=True)
        method = df['Method'].unique().tolist()[0]
        if method in name_map:
            method = name_map[method]
        this_ef = getEF(df)[[EFname, 'Target']]
        this_ef.rename(columns={EFname: method}, inplace=True)
        this_ef = pd.merge(this_ef, ref_EF, on=['Target'])
        this_ef[diffname] = this_ef[method] - this_ef[refmethod]
        this_ef.sort_values(diffname, ascending=False, inplace=True)
        subax = plt.subplot2grid((grid_length,grid_width),
                            (i // grid_width, i % grid_width),
                            fig=fig)
        color = paper_palettes[method] if method in paper_palettes else \
                backup_palette[i % len(backup_palette)]
        sns.barplot(x=refmethod, y="Target", data=this_ef, color=color,
                ax=subax, label=EFname, alpha=0.6)
        sns.barplot(x=diffname, y="Target", data=this_ef, color=color, ax=subax,
                label="Improvement over %s" %refmethod)
        subax.set_title(method.replace('\n',' '))
        # subax.set_xlim((-8,8)) # general
        # subax.set_xlim((-40, 60)) # crossdock-dude
        subax.set_xlim((-10, 15)) # crossdock-litpcba
        if (i // grid_width) != grid_length-1:
            subax.set_xlabel('')
        if (i % grid_width) != 0:
            subax.set_ylabel('')
        sns.despine(left=True, bottom=True)

plt.tight_layout()
fig.legend([black_square, gray_square], ['Improvement over\n%s'%refmethod.replace('\n', ' '),
    '%s\n%s' %(EFname,refmethod.replace('\n', ' '))], frameon=True, 
    bbox_to_anchor=(1.4, 0.25)) # crossdock
    # bbox_to_anchor=(1.3, 0.25)) # general
    # bbox_to_anchor=(0.95, 0.25)) # refined
outrefname = refmethod
if refmethod in reverse_map:
    outrefname = reverse_map[refmethod]
fig.savefig('%sEFdiff_vs_%s.pdf' %(args.outprefix,outrefname), bbox_inches='tight')
