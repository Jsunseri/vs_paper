#!/bin/env python
from argparse import ArgumentParser
import os
import pandas as pd
from plumbum.cmd import uniq,awk,sdsorter

parser = ArgumentParser(description='Make accessory file of features with rows that match a .types file')
parser.add_argument('-t', '--types', nargs='+', help='Input .types file(s) used to order rows')
parser.add_argument('-tc', '--typescols', type=str, default='Label,Affinity,Recfile,Ligfile', 
        help='Comma-separated column identifiers for .types file')
parser.add_argument('-s', '--summary', nargs='+', help='One or more summary '
        'files providing features to extract with accompanying metadata used to '
        'match to .types file')
parser.add_argument('-sc', '--summarycols', type=str, default='Label,Prediction,Target,Title,Method', 
        help='Comma-separated column identifiers for summary file')
parser.add_argument('-sp', '--special_casing', action='store_true', 
        help='Special case how Target and Title fields are determined; '
        '(default is to read actual molecule titles from files). '
        'When special_casing, pass simple descriptor model '
        'predictions as the types file, typescols are ignored')
parser.add_argument('-d', '--data_root', nargs='*', 
        help='Common top-level directory for all data files')
parser.add_argument('-o', '--outprefix', type=str, default='extra_features', 
        help='Prefix for output file of additional features')
args = parser.parse_args()

order = []
if args.special_casing:
    for f in args.types:
        with open(f, 'r') as ifile:
            for line in ifile:
                contents = line.split()
                target = contents[2]
                title = contents[3]
                order.append((target,title))
else:
    # read in with pandas
    # the .types files don't have molecular titles. since this script is being used
    # together with simple descriptor model training, i'm adopting the conventions i use there -
    # i.e. no gninatypes, if a file is multi-model, i repeat the filename on
    # contiguous lines for each example from the file
    # TODO for now just handle multi-model smi and sdf if we aren't special-casing
    unique_ligs = []
    for f in args.types:
        unique_ligs += (awk['{print $NF}', f] | uniq)().strip().split('\n')
    
    for lig in unique_ligs:
        if args.data_root:
            found = False
            for path in args.data_root:
                fullname = os.path.join(path, lig)
                if os.path.isfile(fullname):
                    found = True
                    break
            assert found, '%s does not exist in any user-provided directories.' %lig
        else:
            fullname = lig
        target = os.path.basename(os.path.dirname(fullname))
        # if it's a smi just read the file, otherwise use sdsorter or error
        base,ext = os.path.splitext(fullname)
        if ext == '.gz':
            base,ext = os.path.splitext(base)
        if ext == '.sdf':
            # sdsorter
            moltitles = (sdsorter['-print', '-omit-header', fullname] | awk['{print $2}'])().strip().split('\n')
        elif ext == '.smi' or ext == '.ism':
            # just read
            with open(fullname, 'r') as f:
                moltitles = [line.strip().split()[-1] for line in f]
        else:
            assert 0, 'Unrecognized extension %s' %ext
        order += [(target,title) for title in moltitles]

# match TARGET TITLE, preserving order of types file
features = {}
for summaryfile in args.summary:
    features[summaryfile] = {}
    with open(summaryfile, 'r') as f:
        for line in f:
            contents = line.strip().split()
            prediction = contents[1]
            target = contents[2]
            title = contents[3]
            features[summaryfile][(target,title)] = prediction

with open('%s.csv' %args.outprefix, 'w') as f:
    for tup in order:
        for fname in args.summary:
            if tup not in features[fname]:
                # the .smi files in dkoes' DUDe directory have the "ZIN" characters
                # in ZINC compound identifiers cut off, for some unknown reason,
                # but the correct titles appear in the sdf files
                alttup = (tup[0],'ZIN' + tup[1])
                if alttup not in features[fname]:
                    print('%s not found in %s, setting output to 0.0' %(tup,fname))
                    features[fname][tup] = '0.0'
                else:
                    features[fname][tup] = features[fname][alttup]
        f.write('%s\n' %(' '.join([features[fname][tup] for fname in args.summary])))
