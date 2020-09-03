#!/bin/env python
import os
import pandas as pd
import numpy as np
from train_simpledescriptor_models import unique_list
from argparse import ArgumentParser
from openbabel import pybel
from plumbum.cmd import sdsorter

parser = ArgumentParser(description='Use two files to make a list of moltitles '
        'that are missing in one file but present in the other')
parser.add_argument('-r', '--rootdir', default='', type=str, help='Optionally '
        'specify root directory for input files')
parser.add_argument('-s', '--summaryfile', required=True, type=str,
        help='Summary file organized LABEL PREDICTION TARGET TITLE METHOD; '
        'titles that do not appear here but appear in the types file will be '
        'added to the exclude list')
parser.add_argument('-t', '--types', required=True, type=str, help='Types file '
        'that, paired with rootdir if necessary, specifies path to files we can '
        'get mol titles from; ligand file path should be the third column')
parser.add_argument('-o', '--out', default='', help='Give an output name, '
        'othrwise one will be inferred from the summary file name')
args = parser.parse_args()

# only these mol titles should be included
df = pd.read_csv(args.summaryfile, delim_whitespace=True, header=None,
        names=['Label', 'Prediction', 'Target', 'Title', 'Method'],
        dtype={'Label': np.float64, 'Prediction': np.float64, 'Target': str,
            'Title': str, 'Method': str})

mols = df['Title'].tolist()
targets = df['Target'].tolist()

# rename the DUDe ZINC compounds that in dkoes' smiles have the ZIN stripped off
dataset = os.path.splitext(os.path.basename(args.types))[0]
isdude = True if dataset == 'dude' else False
if isdude:
    for i,mol in enumerate(mols):
        newmol = mol.lstrip('ZIN')
        if mol != newmol:
            mols[i] = newmol

include = zip(mols, targets)
# get all moltitles associated with types
molfiles = pd.read_csv(args.types, delim_whitespace=True, header=None,
        names=['Label', 'Recfile', 'Ligfile'])['Ligfile'].tolist()
molfiles = unique_list(molfiles)
moltitles = []
moltargets = []
for molname in molfiles:
    target = os.path.basename(os.path.dirname(molname))
    _,ext = os.path.splitext(molname)
    ext = ext.split('.')[-1]
    if ext == 'ism': ext = 'smi'
    assert ext == 'smi' or ext == 'sdf', 'Only smi or sdf supported in types file'
    fullname = os.path.join(args.rootdir, molname)
    if ext == 'smi':
        with open(fullname, 'r') as f:
          these_mols = [line.strip().split()[-1] for line in f]
          moltitles += these_mols
    elif ext == 'sdf':
        these_mols = (sdsorter['-print', '-omit-header', fullname] | awk['{print $2}'])().strip().split('\n')
        moltitles += these_mols
    moltargets += [target] * len(these_mols)
allmols = zip(moltitles, moltargets)

include = set(include)
# write out anything in the types that doesn't appear in the input summary file
if args.out:
    fname = args.out
else:
    base = os.path.splitext(os.path.basename(args.summaryfile))[0]
    fname = '%s_excludelist' %(base)
with open(fname, 'w') as e:
    for name in allmols:
        if name not in include:
            e.write('%s %s\n' %(name[1], name[0]))
