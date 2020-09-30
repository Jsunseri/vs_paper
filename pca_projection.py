#!/bin/env python
import os,sys,math
from argparse import ArgumentParser

import matplotlib as mpl
mpl.use('TKAgg')
import matplotlib.pyplot as plt
from vspaper_settings import backup_palette

from sklearn.decomposition import PCA
import numpy as np
from openbabel import pybel

from plumbum.cmd import wc

# need it for negative signs it turns out
mpl.rcParams.update({'text.usetex': 'True'})
parser = ArgumentParser(description='Do PCA on one dataset, plot the projection '
        'onto the first two principal components for that and any additional '
        'datasets')
parser.add_argument('-p', '--primary', required=True, type=str, help='Dataset '
        'used for PCA and name used to label it in the figure, separated by a '
        'comma')
parser.add_argument('-a', '--additional', nargs='*', default=[],
        help='Additional datasets to project onto components and plot and '
        'labels, also comma-separated')
parser.add_argument('-o', '--outprefix', default='', help='Prefix for output')
args = parser.parse_args()

primary_dataset,primary_label = args.primary.split(',')
ext = os.path.splitext(primary_dataset)[-1]
assert ext == '.smi' or ext == '.ism', 'Only reading from smiles is supported'
for item in args.additional:
    fname,_ = item.split(',')
    ext = os.path.splitext(fname)[-1]
    assert ext == '.smi' or ext == '.ism', 'Only reading from smiles is supported'

nmols = int((wc['-l', primary_dataset])().split()[0])
# N_samples x N_features
X = np.zeros((nmols, 1024))
failures = []
with open(primary_dataset, 'r') as f:
    for i,line in enumerate(f):
        smi = line.split()[0]
        pieces = smi.split('.')
        if len(pieces) > 1:
            smi = max(pieces, key=len)
        mol = pybel.readstring('smi', smi)
        if mol is not None:
            mol.addh()
            fp = mol.calcfp('ecfp4')
            ovec = ['1' if j in fp.bits else '0' for j in range(1024)]
            X[i,:] = np.asarray(ovec)
        else:
            failures.append(i)

if failures:
    np.delete(X, failures, axis=0)
pca = PCA(n_components=2)
pca.fit(X)
X_r = pca.transform(X)

total_plots = len(args.additional) + 1
grid_width = int(math.ceil(math.sqrt(total_plots)))
grid_length = int(math.ceil(float(total_plots)/grid_width))
plot_num = 0
fig,ax = plt.subplots()
sub_ax = plt.subplot2grid((grid_length,grid_width),
        (plot_num // grid_width, plot_num % grid_width),
        fig=fig)
sub_ax.scatter(X_r[:,0], X_r[:,1], color=backup_palette[0], alpha=0.6,
        label=primary_label.replace('_', ' '), lw=2)
sub_ax.set_xlabel('PC1')
sub_ax.set_ylabel('PC2')
sub_ax.legend(frameon=True)

for num,item in enumerate(args.additional):
    contents = item.split(',')
    fname = contents[0]
    label = contents[1]
    nmols = int((wc['-l', fname])().split()[0])
    X2 = np.zeros((nmols, 1024))
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
    plot_num = num + 1
    X2_r = pca.transform(X2)
    color_idx = (i+1) % len(backup_palette)
    sub_ax = plt.subplot2grid((grid_length,grid_width),
            (plot_num // grid_width, plot_num % grid_width),
            fig=fig)
    sub_ax.scatter(X2_r[:,0], X2_r[:,1], color=backup_palette[color_idx],
            alpha=0.6, label=label.replace('_', ' '), lw=2)
    sub_ax.set_xlabel('PC1')
    sub_ax.set_ylabel('PC2')
    sub_ax.legend(frameon=True)

fig.savefig('%sPCA_projection.png' %args.outprefix, dpi=300, bbox_inches='tight')
