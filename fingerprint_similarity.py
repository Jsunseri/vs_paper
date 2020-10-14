#!/bin/env python
import sys,os
from argparse import ArgumentParser

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

from rdkit.Chem import AllChem
from rdkit.DataStructs import FingerprintSimilarity as fs
from rdkit.Chem.Fingerprints import FingerprintMols

import vspaper_settings

# pass in smiles, make histplot of Tanimoto sim based on ECFP4 fingerprints
parser = ArgumentParser(description='Plot histogram of Tanimoto similarity of '
        'fingerprints for input smiles')
parser.add_argument('-f', '--files', nargs='+', help='Smiles files')
parser.add_argument('-o', '--outprefix', default='fps', help='Prefix for output fig')
args = parser.parse_args()

fingerprints = []
for fname in args.files:
    base,ext = os.path.splitext(fname)
    ext = ext.split('.')[-1]
    assert (ext == 'smi') or (ext == 'ism'), 'Only smi supported'
    with open(fname, 'r') as f:
        for line in f:
            smi = line.split()[0]
            mol = AllChem.MolFromSmiles(smi)
            if mol == None:
                mol = AllChem.MolFromSmiles(smi,sanitize=False)
            fingerprints.append(FingerprintMols.FingerprintMol(mol))

sims = [0] * len(fingerprints)
for i,fp in enumerate(fingerprints):
    for j in range(len(fingerprints)):
        if i == j:
            continue
        fpj = fingerprints[j]
        sim = fs(fp,fpj)
        if sim > sims[i]:
            sims[i] = sim

fig,ax = plt.subplots()
sns.histplot(sims, ax=ax, color='#8ed74f', edgecolor='white')
ax.set_xlabel('Max Similarity')
ax.set_xlim(0, 1.0)
fig.canvas.draw()
labels = [item.get_text() for item in ax.get_xticklabels()]
labels[0] = ''
ax.set_xticklabels(labels)
fig.savefig('%shistplot.pdf' %args.outprefix, bbox_inches='tight')
