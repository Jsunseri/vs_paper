#!/bin/env python
from argparse import ArgumentParser
import os
import pandas as pd
from plumbum.cmd import uniq,awk,sdsorter

parser = ArgumentParser(description='Make accessory file of features with rows that match a .types file')
parser.add_argument('-t', '--types', type=str, required=True, help='Input .types file used to order rows')
parser.add_argument('-tc', '--typescols', type=str, default='Label,Affinity,Recfile,Ligfile', 
        help='Comma-separated column identifiers for .types file')
parser.add_argument('-s', '--summary', nargs='+', help='One or more summary '
        'files providing features to extract with accompanying metadata used to '
        'match to .types file')
parser.add_argument('-sc', '--summarycols', type=str, default='Label,Prediction,Target,Title,Method', 
        help='Comma-separated column identifiers for summary file')
parser.add_argument('-d', '--data_root', type=str, default='', 
        help='Common top-level directory for all data files')
args = parser.parse_args()

# read in with pandas
# the .types files don't have compound titles. since this script is being used
# along with random forest training, i'm adopting the conventions i use there -
# i.e. no gninatypes, if a file is multi-model, i repeat the filename on
# contiguous lines for each example from the file
# TODO ok, for now just handle multi-model smi and sdf
unique_ligs = (awk['{print $NF}', arg.types] | uniq)().strip().split('\n')
unique_ligs = [os.path.join(args.data_root,x) for x in unique_ligs]
for lig in unique_ligs:
    # if it's a smi just read the file, otherwise use sdsorter or error
    base,ext = os.path.splitext(lig)
    if ext == '.gz':
        base,ext = os.path.splitext(base)
    if ext == '.sdf':
        # sdsorter
    elif ext == '.smi' or ext == '.ism':
        # just read
    else:
        assert 0, 'Unrecognized extension %s' %ext

typesfile = pd.read_csv(args.types, delim_whitespace=True, columns=args.typescols)

for summaryfile in args.summary:
    summarydf = pd.read_csv(summaryfile, delim_whitespace=True, columns=args.summarycols)
