#!/bin/env python
import sys,os
from openbabel import pybel
import pandas as pd
from argparse import ArgumentParser

# you give us a types file, we dump out plaintext fingerprints in a Nmols X
# Nbits csv (yes, for real, i know, if it's bad i'll be motivated to change it)
parser = ArgumentParser(description='Dump csv of fingerprints for mols in input types')
parser.add_argument('-r', '--root', nargs='*', 
        help='Root directory to complete path to files in types; pass more than '
        'one and if we fail to find a file in the first location we can try the '
        'subsequent ones as backups')
parser.add_argument('-f', '--files', nargs='+', help='Types files')
parser.add_argument('-k', '--keep_first', action='store_true', help='If a '
        'structure file has multiple mols, only keep the first one')
parser.add_argument('-o', '--outprefix', default='fps', help='Prefix for output csv')
args = parser.parse_args()

ligs = []
for f in args.files:
    ligs += pd.read_csv(f, delim_whitespace=True, header=None,
            names=['Score', 'Rec', 'Lig'])['Lig'].unique().tolist()

with open(args.outprefix+'.csv', 'w') as ofile:
    for fname in ligs:
        base,ext = os.path.splitext(fname)
        ext = ext.split('.')[-1]
        found = False
        for dname in args.root:
            fullpath = os.path.join(dname, fname)
            if os.path.isfile(fullpath):
                found = True
                break
        assert found, '%s not found in any user-provided directories' %fname
        if ext == 'smi' or ext == 'ism':
            with open(fullpath, 'r') as f:
                for line in f:
                    smi = line.split()[0]
                    m = pybel.readstring('smi', smi)
                    fp = m.calcfp('ecfp4')
                    ovec = ['1' if i in fp.bits else '0' for i in range(1024)]
                    ofile.write(' '.join(ovec) + '\n')
                    if args.keep_first:
                        break
        else:
            try:
                mols = pybel.readfile(ext, fullpath)
                for m in mols:
                    fp = m.calcfp('ecfp4')
                    ovec = ['1' if i in fp.bits else '0' for i in range(1024)]
                    ofile.write(' '.join(ovec) + '\n')
                    if args.keep_first:
                        break
            except Exception as e:
                print(e)
                # I'm assuming this was a user error so I'm choosing not to
                # handle anything
                sys.exit('Filetype %s not supported' %ext)
