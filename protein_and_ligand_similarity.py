#!/bin/env python
import os,glob,pickle
from statistics import mean,median
from argparse import ArgumentParser
from multiprocessing import Pool, cpu_count
from functools import partial

from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.Polypeptide import three_to_one
from Bio.PDB.Polypeptide import is_aa
from Bio import pairwise2

from openbabel import pybel

import matplotlib.pyplot as plt
import seaborn as sns

import vspaper_settings

def getLigandPrints(flist):
    '''
    Get list of ligand fingerprints 
    '''
    fingerprints = []
    names = []
    for fname in flist:
        base,ext = os.path.splitext(fname)
        ext = ext.split('.')[-1]
        if ext == 'smi' or ext == 'ism':
            with open(fname, 'r') as f:
                for line in f:
                    contents = line.split()
                    smi = contents[0]
                    name = contents[1]
                    m = pybel.readstring('smi', smi)
                    fingerprints.append(m.calcfp('ecfp4'))
                    names.append(name)
        else:
            try:
                mols = pybel.readfile(ext, fname)
                for m in mols:
                    fingerprints.append(m.calcfp('ecfp4'))
            except Exception as e:
                print(e)
    return (fingerprints,names)

def getMaxLigandSim(train, test):
    '''
    Get the max ligand sim for the test ligands among the train ligands
    '''
    sims = [0] * len(test)
    for i,fpi in enumerate(test):
        for j,fpj in enumerate(train):
            sim = fpi | fpj
            if sim > sims[i]:
                sims[i] = sim
    return sims

def getLigs(reflist, include_inactive=False):
    '''
    Given a list of reference proteins, get all relevant lig fingerprints
    '''
    # this is going to be special-cased and crappy
    activelist = [[],[]]
    decoylist = [[],[]]
    for ref in reflist:
        dirname = os.path.dirname(ref)
        dirstr = os.path.basename(os.path.dirname(dirname))
        if ('general' in dirstr) or ('refined' in dirname):
            # either general or refined
            base = ref.split('_protein.pdb')[0]
            name = base + '_ligand.smi'
            if not os.path.isfile(name):
                name = base + '_ligand.sdf'
                if not os.path.isfile(name):
                    name = base + '_ligand.mol2'
            assert os.path.isfile(name), '%s expected from %s but not found' %(name, ref)
            out = getLigandPrints([name])
            activelist[0] += out[0]
            activelist[1] += out[1]
        elif 'DUDe' in dirstr:
            # DUD-E
            name = os.path.join(dirname, 'actives_final.ism')
            assert os.path.isfile(name), '%s expected from %s but not found' %(name, ref)
            out = getLigandPrints([name])
            activelist[0] += out[0]
            activelist[1] += out[1]
            if include_inactive:
                name = os.path.join(dirname, 'decoys_final.ism')
                assert os.path.isfile(name), '%s expected from %s but not found' %(name, ref)
                out = getLigandPrints([name])
                decoylist[0] += out[0]
                decoylist[1] += out[1]
        elif 'lit-pcba' in dirstr:
            # lit-pcba
            names = glob.glob(dirname + '/AID*_active.smi')
            assert len(names) == 1, 'Problem with %s' %ref
            out = getLigandPrints(names)
            activelist[0] += out[0]
            activelist[1] += out[1]
            if include_inactive:
                names = glob.glob(dirname + '/AID*_inactive.smi')
                assert len(names) == 1
                out = getLigandPrints(names)
                decoylist[0] += out[0]
                decoylist[1] += out[1]
        elif 'Pocketome' in dirstr:
            # cross-docked, there really can be more than one file here
            names = glob.glob(dirname + '/*_lig.pdb')
            assert len(names) > 0, 'Problem with %s' %ref
            out = getLigandPrints(names)
            activelist[0] += out[0]
            activelist[1] += out[1]
        else:
            assert False, 'Failed to find ligs for %s' %ref
    return (activelist, decoylist)

def calcDist(refs, target):
    '''
    compute min distance between target and all refs
    '''
    # list of seqs for this target from the test set
    a = target[1]
    simlist = []
    mindist = 0.5
    # compare this target from the test set to all targets in the train set
    for (name,b) in refs:
        # check distance for all seqs associated with each target
        for seq1 in a:
            for seq2 in b:
                score = pairwise2.align.globalxx(seq1, seq2, score_only=True)
                length = max(len(seq1), len(seq2))
                distance = (length-score)/length
                if distance < mindist:
                    simlist.append(name)
                    continue
    return (target[0], simlist)

def getAllLigs(targets, refs):
    '''
    for each target in targets, get the ligands associated with all proteins 
    in ref that have >0.5 sequence identity (for cross-docked this includes
    any ligand in the pocket)
    '''
    print('calculating protein distances...')
    pool = Pool()
    function = partial(calcDist, refs)
    target_lists = pool.map(function, targets)

    # get all ligands in the pockets associated with those training set receptors
    # we don't include inactives here
    outlist = {}
    for targetpath, reflist in target_lists:
        print('calculating training ligand fingerprints...')
        outlist[targetpath] = getLigs(reflist)[0]
    return outlist

def getResidueStrings(structure):
    seqs = []
    for model in structure:
        for ch in model.get_chains():
            seq = ''
            for residue in model.get_residues():
                resname = residue.get_resname()
                if is_aa(resname, standard=True):
                    seq += three_to_one(resname)
                elif resname in {'HIE', 'HID'}:
                    seq += 'H'
                elif resname in {'CYX', 'CYM'}:
                    seq += 'C'
                else:
                    seq += 'X'
            seqs.append(seq)
    return seqs

def readPDBfiles(pdbfiles,ncpus=cpu_count()):
    pdb_parser = PDBParser(PERMISSIVE=1, QUIET=1)
    with open(pdbfiles, 'r') as f:
        pdblines = set(f.readlines())
    pool = Pool(ncpus)
    function = partial(loadTarget, pdb_parser)
    target_tups = pool.map(function, pdblines)
    return target_tups

def loadTarget(pdb_parser, line):
    target_pdb = line.strip()
    target_name = os.path.splitext(os.path.basename(line))[0].split('_')[0]
    if target_name == 'receptor':
        target_name = os.path.basename(os.path.dirname(line))
    try:
        structure = pdb_parser.get_structure(target_name, target_pdb)
        seqs = getResidueStrings(structure)
        # full path to file, list of string seqs generated from models * chains
        return (target_pdb, seqs)
    except IOError as e:
        print(e)

if __name__ == '__main__':
    parser = ArgumentParser(description='Use both proteins and the '
            'ligands associated with them to compute similarity between two datasets')
    parser.add_argument('-t','--testfile',type=str,help="input file with paths to proteins for testing")
    parser.add_argument('-r','--reffile',type=str,help="file with paths to reference proteins for distance calculation")
    args = parser.parse_args()

    print('reading pdbs...')
    # readPDBfiles returns list of (path_to_target_file, target_seqs) tuples
    testset = readPDBfiles(args.testfile)
    testbase = os.path.splitext(os.path.basename(args.testfile))[0]
    refset = readPDBfiles(args.reffile)
    refbase = os.path.splitext(os.path.basename(args.reffile))[0]

    # find all ligs in the train set associated with targets that are less
    # than 0.5 away from the test set protein, this returns a dict 
    # that maps path_to_test_target to 
    # (list_of_relevant_train_lig_fps, list_of_relevant_train_lig_names)
    protein_train_lig_list = getAllLigs(testset, refset)

    outprefix = '%s_to_%s' %(testbase, refbase)
    
    # build up the ligand list (including inactives if applicable) for the
    # test protein
    # then find the max similarity to the training set ligands for each ligand in the
    # test set for this protein - do this for actives AND inactives, but keep the
    # results separate
    print('calculating test ligand fingerprints')
    numtests = len(testset)
    with open("%s_combined_sims.csv" %(outprefix), "w") as f:
        f.write("Target Active_Mean Active_Median Inactive_Mean Inactive_Median\n")
        for testnum,test in enumerate(testset):
            path = test[0]
            target_name = os.path.basename(os.path.dirname(path))
            print('target %d/%d' %(testnum+1,numtests))
            # returns tup(([active_fps],[active_names]), ([decoy_fps],[decoy_names]))
            activelist, decoylist = getLigs([path], True)
            active_sims = getMaxLigandSim(protein_train_lig_list[path][0],activelist[0])
            decoy_sims = getMaxLigandSim(protein_train_lig_list[path][0],decoylist[0])
            # dump out the means and medians per target for this dataset, to be used for
            # plotting with other datasets, format like 
            # TEST_TARGET ACTIVE_MEAN ACTIVE_MEDIAN INACTIVE_MEAN INACTIVE_MEDIAN
            # also dump out the max similarity per ligand into a separate file
            f.write('%s %0.3f %0.3f %0.3f %0.3f\n' %(target_name,
                mean(active_sims), median(active_sims), mean(decoy_sims),
                median(decoy_sims)))
            with open('%s_%s_active_maxsim.csv' %(target_name,outprefix), 'w') as g:
                for sim,name in zip(active_sims, activelist[1]):
                    g.write('%s %0.3f\n' %(name,sim))
