#!/bin/env python
import os,sys,math
from argparse import ArgumentParser

import matplotlib as mpl
mpl.use('TKAgg')
import matplotlib.pyplot as plt

from vspaper_settings import backup_palette
from train_simpledescriptor_models import get_muv_descriptors_babel

from sklearn.decomposition import PCA
import numpy as np
from openbabel import pybel

from plumbum.cmd import wc

# need it for negative signs it turns out
mpl.rcParams.update({'text.usetex': 'True'})
parser = ArgumentParser(description='Do PCA on one dataset, plot the projection '
        'onto the first two principal components for that and any additional '
        'datasets')
parser.add_argument('-p', '--primary', nargs='+', help='Datasets '
        'used for PCA and names used to label them in the figure, separated by '
        'commas')
parser.add_argument('-a', '--additional', nargs='*', default=[],
        help='Additional datasets to project onto components and plot and '
        'labels, also comma-separated')
parser.add_argument('-d', '--descriptors', default='', help='Optionally specify '
        'descriptor set to define the feature vector instead of using '
        'fingerprints - choices are "DUD-E", "MUV", or "all"')
parser.add_argument('-z', '--zorder', type=int, default=0, help='Optionally '
        'set zorder for the "primary" dataset; by default it will have the '
        'lowest zorder')
parser.add_argument('-o', '--outprefix', default='', help='Prefix for output')
args = parser.parse_args()

filesizes = {}
totalmols = 0
primary_data = []
for primary in args.primary:
    primary_dataset,primary_label = primary.split(',')
    primary_data.append((primary_dataset,primary_label))
    ext = os.path.splitext(primary_dataset)[-1]
    assert ext == '.smi' or ext == '.ism', 'Only reading from smiles is supported'
    
    nmols = int((wc['-l', primary_dataset])().split()[0])
    filesizes[primary_dataset] = nmols
    totalmols += nmols

for item in args.additional:
    fname,_ = item.split(',')
    ext = os.path.splitext(fname)[-1]
    assert ext == '.smi' or ext == '.ism', 'Only reading from smiles is supported'

if args.descriptors:
    assert (args.descriptors == 'DUD-E') or (args.descriptors == 'MUV') or (args.descriptors == 'all'), ('%s '
    'not a recognized descriptor class' %args.descriptors)
    if args.descriptors == 'DUD-E':
        nfeatures = 6
    elif args.descriptors == 'MUV':
        nfeatures = 17
    elif args.descriptors == 'all':
        nfeatures = 20
else:
    nfeatures = 1024

# N_samples x N_features
X = np.zeros((totalmols, nfeatures))
failures = []
offsets = [0]
count = 0
for data_idx,(primary_dataset,primary_label) in enumerate(primary_data):
    offset = offsets[data_idx]
    with open(primary_dataset, 'r') as f:
        for i,line in enumerate(f):
            idx = offset + i
            smi = line.split()[0]
            pieces = smi.split('.')
            if len(pieces) > 1:
                smi = max(pieces, key=len)
            mol = pybel.readstring('smi', smi)
            if mol is not None:
                mol.addh()
                if args.descriptors == 'DUD-E':
                    desc = mol.calcdesc()
                    desc['charge'] = mol.charge
                    ovec = [desc[d] for d in ['MW','HBA1','HBD','rotors','logP','charge']]
                elif args.descriptors == 'MUV' or args.descriptors == 'all':
                    ovec = get_muv_descriptors_babel(mol)
                    if args.descriptors == 'all':
                        desc = mol.calcdesc()
                        desc['charge'] = mol.charge
                        ovec += [desc[d] for d in ['MW', 'rotors', 'charge']]
                else:
                    fp = mol.calcfp('ecfp4')
                    ovec = ['1' if j in fp.bits else '0' for j in range(1024)]
                X[idx,:] = np.asarray(ovec)
            else:
                failures.append(idx)
            count += 1
    expected_count = filesizes[primary_dataset] + offset
    assert count == expected_count, ('Got %d mols so far but was expecting %d' %(count, expected_count))
    offsets.append(count)
    
if failures:
    np.delete(X, failures, axis=0)

pca = PCA(n_components=2)
pca.fit(X)
X_r = pca.transform(X)

fig,ax = plt.subplots()
for idx,(primary_dataset,primary_label) in enumerate(primary_data):
    color_idx = idx % len(backup_palette)
    start = offsets[idx]
    stop = start + filesizes[primary_dataset]
    ax.scatter(X_r[start:stop,0], X_r[start:stop,1], color=backup_palette[color_idx], alpha=0.6,
            label=primary_label.replace('_', ' '), lw=2,
            zorder=args.zorder+idx)

for num,item in enumerate(args.additional):
    contents = item.split(',')
    fname = contents[0]
    label = contents[1]
    nmols = int((wc['-l', fname])().split()[0])
    X2 = np.zeros((nmols, nfeatures))
    failures = []
    with open(fname, 'r') as f:
        for i,line in enumerate(f):
            smi = line.split()[0]
            pieces = smi.split('.')
            if len(pieces) > 1:
                smi = max(pieces, key=len)
            mol = pybel.readstring('smi', smi)
            if mol is not None:
              mol.addh()
              if args.descriptors == 'DUD-E':
                  desc = mol.calcdesc()
                  desc['charge'] = mol.charge
                  ovec = [desc[d] for d in ['MW','HBA1','HBD','rotors','logP','charge']]
              elif args.descriptors == 'MUV' or args.descriptors == 'all':
                  ovec = get_muv_descriptors_babel(mol)
                  if args.descriptors == 'all':
                      desc = mol.calcdesc()
                      desc['charge'] = mol.charge
                      ovec += [desc[d] for d in ['MW', 'rotors', 'charge']]
              else:
                  fp = mol.calcfp('ecfp4')
                  ovec = ['1' if j in fp.bits else '0' for j in range(1024)]
              try:
                  X2[i,:] = np.asarray(ovec)
              except Exception as e:
                  print(e)
                  print(X2.shape)
                  print(fname)
                  sys.exit()
            else:
                failures.append(i)
    if failures:
        np.delete(X2, failures, axis=0)
    X2_r = pca.transform(X2)
    fullnum = len(primary_data) + num
    color_idx = fullnum % len(backup_palette)
    ax.scatter(X2_r[:,0], X2_r[:,1], color=backup_palette[color_idx],
            alpha=0.6, label=label.replace('_', ' '), lw=2)

ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.legend(frameon=True)

fig.savefig('%sPCA_projection.png' %args.outprefix, dpi=300, bbox_inches='tight')
