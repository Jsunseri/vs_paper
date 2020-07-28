#!/bin/env python
import sys,os,re,glob,gzip
from argparse import ArgumentParser
from collections import namedtuple
from joblib import dump

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

from rdkit import Chem
from rdkit.Chem import QED
from rdkit.Chem import rdmolops
from rdkit.Chem import rdMolDescriptors

FoldData = namedtuple('FoldData', ['fnames', 'labels', 'fold_it'])
FeaturizeOutput = namedtuple('FeaturizeOutput', ['features', 'failures'])
FixedOutput = namedtuple('FixedOutput', ['labels', 'fold_it'])

def unique_list(seq):
    '''
    Return new list with just the unique elements from the input, in the same
    order as they first appeared in the input
    '''
    seen = set()
    return [x for x in seq if x not in seen and not seen.add(x)]

def exclusive_scan(iterable):
    '''
    Generate exclusive scan from an iterator
    '''	
    total = 0
    yield total
    for value in iterable:
    	total += value
    	yield total 

def find_and_parse_folds(prefix, foldnums='', columns='Label,Affinity,Recfile,Ligfile', fit_all=False):
    '''
    Parameters
    ----------
    prefix: str
        prefix for the training fold filenames
    foldnums: str, optional
        which foldnums to include; default is to glob and include all matches
    columns: str, optional
        comma-separated list of column identifiers for fold files; defaults to
        "Label,Affinity,Recfile,Ligfile"
    fit_all: bool, optional
        whether to fit all numerical columns with RandomForestRegressor.
        default is to prefer affinity if available, fit classifier with label
        if it is not, and complain otherwise unless this was passed

    Returns
    ----------
    fnames: array_like
        molecule filenames
    labels: array_like
        y values
    fold_it: iterable
        iterable yielding fold indices within fnames and labels
    '''
    # identify the fold files; we'll use the same typefile format as used for
    # gnina. since we assume cross validation is being performed, we just parse
    # the _test_ folds, and set up train/test indices for cross validation
    # accordingly
    files = []
    if not foldnums:
        foldnums = set()
        glob_files = glob.glob(prefix + '*')
        pattern = r'(%s)(test)(\d+)\.types$' % (prefix)
        for file in glob_files:
            match = re.match(pattern, file)
            if match:
                foldnums.add(int(match.group(3)))
        foldnums = list(foldnums)
    else:
        foldnums = [int(i) for i in foldnums.split(',') if i]
    foldnums.sort()
    for i in foldnums:
        files.append('%stest%d.types' % (prefix, i))

    # we *require* a ligfile and at least one target for
    # regression/classification. by default we look for label and affinity; if
    # only a label is present we use it for classification, and if both are
    # present we use just the affinity for regression. sklearn doesn't seem to
    # allow mixed-task multi-task models, so if --fit_all is passed, we train a
    # multitask regressor on all numerical columns (including the label if
    # present)
    column_names = [name for name in columns.split(',') if name]
    assert column_names, "Missing column names for types files."
    assert 'Label' or 'Affinity' in column_names, "Currently a 'Label' or 'Affinity' column are required."
    assert 'Ligfile' in column_names, "Path to ligand file required."

    # parse files with pandas; since we have allowed the column layout to vary,
    # we default to the "normal" version but allow the user to specify others.
    df_list = [pd.read_csv(fname, names=column_names, delim_whitespace=True) for fname in files]

    # fill in list of tuples arranged like [(train0, test0), (train1, test1), ..., (trainN, testN)]
    # where the elements are arrays of indices associated with these folds
    elems_per_df = [df.shape[0] for df in df_list]
    total_examples = sum(elems_per_df)
    print('Got %d examples from %d files' %(total_examples, len(files)))

    def get_indices(numlist, idx):
        '''
        return (train,test) indices, where numlist is the number of examples
        associated with each test fold, idx is the test fold for this round and
        the other test folds will be concatenated to constitute the train set
        '''
        train = []
        for i,num in enumerate(numlist):
            start = 0 if i==0 else numlist[i-1]
            if i != idx:
                train += list(range(start, start+num))
            else:
                test = list(range(start, start+num))
        return (train,test)

    fold_it = [get_indices(elems_per_df, idx) for idx in range(len(elems_per_df))]

    # get lig filenames
    df = pd.concat(df_list, ignore_index=True, sort=False)
    fnames = df['Ligfile'].to_numpy()

    # get y_values
    allcols = df.columns
    ycols = []
    if fit_all:
    	for col in allcols:
    	    if is_numeric_dtype(df[col]):
    	        ycols.append(col)
    else:
        if 'Affinity' in allcols:
            ycols = ['Affinity']
        elif 'Label' in allcols:
            ycols = ['Label']
        else:
            assert 0, 'Custom fit targets not implemented yet. Pass --fit_all '
            'or specify target column as "Affinity"'
    labels = df[ycols].to_numpy()
    return FoldData(fnames, labels, fold_it)

def get_dude_descriptors(mol):
    '''
    Parameters
    ----------
    mol: object
        rdkit mol object

    Returns
    ----------
    dude_descriptors: array_like
        array of the calculated descriptors for mol. descriptors used are the
        same as those used for construction of DUD-E. they are:
        
            + molecular weight
            + number of hydrogen bond acceptors
            + number of hydrogen bond donors
            + number of rotatable bonds
            + logP 
            + net charge
    '''

    props = QED.properties(mol)
    dude_descriptors = []
    dude_descriptors.append(props.MW)
    dude_descriptors.append(props.HBA)
    dude_descriptors.append(props.HBD)
    dude_descriptors.append(props.ROTB)
    dude_descriptors.append(props.ALOGP)
    dude_descriptors.append(rdmolops.GetFormalCharge(mol))

    return dude_descriptors

def get_muv_descriptors(mol):
    '''
    Parameters
    ----------
    mol: object
        rdkit mol object

    Returns
    ----------
    muv_descriptors: array_like
        array of the calculated descriptors for mol. descriptors used are the
        same as those used for construction of MUV. they are:

            + number of hydrogen bond acceptors
            + number of hydrogen bond donors
            + logP
            + number of all atoms 
            + number of heavy atoms 
            + number of boron atoms 
            + number of bromine atoms 
            + number of carbon atoms 
            + number of chlorine atoms 
            + number of fluorine atoms 
            + number of iodine atoms
            + number of nitrogen atoms 
            + number of oxygen atoms
            + number of phosphorus atoms 
            + number of sulfur atoms 
            + number of chiral centers 
            + number of ring systems
    '''

    props = QED.properties(mol)
    muv_descriptors = []
    muv_descriptors.append(props.HBA)
    muv_descriptors.append(props.HBD)
    muv_descriptors.append(props.ALOGP)
    muv_descriptors.append(mol.GetNumAtoms(onlyExplicit=False))
    muv_descriptors.append(mol.GetNumHeavyAtoms())
    type_list = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    atom_counts = pd.DataFrame(type_list, columns=['type']).groupby('type').size().to_dict()

    muv_descriptors.append(atom_counts[5])  # boron
    muv_descriptors.append(atom_counts[35]) # bromine
    muv_descriptors.append(atom_counts[6])  # carbon
    muv_descriptors.append(atom_counts[17]) # chlorine
    muv_descriptors.append(atom_counts[9])  # fluorine
    muv_descriptors.append(atom_counts[53]) # iodine
    muv_descriptors.append(atom_counts[7])  # nitrogen
    muv_descriptors.append(atom_counts[8])  # oxygen
    muv_descriptors.append(atom_counts[15]) # phosphorus
    muv_descriptors.append(atom_counts[16]) # sulfur
    
    # TODO: will this actually get them all?
    chiral_centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
    muv_descriptors.append(len(chiral_centers))
    muv_descriptors.append(rdMolDescriptors.CalcNumRings(mol))

    return muv_descriptors

# we generate descriptors with rdkit and give the user the option of using the
# DUD-E or MUV descriptors
def generate_descriptors(mol_list, data_root='', method='DUD-E'):
    '''
    Parameters
    ----------
    mol_list: array_like
        list of molecule filenames
    data_root: str
        common path to join with molnames to generate full location
    method: {'DUD-E', 'MUV'}, optional
        Which set of descriptors to use; either DUD-E (the default) or MUV

    Returns
    ----------
    features: array_like
        N_mols X N_features array of standardized features
    failures: array_like
        indices of mols we failed to parse
    '''

    # support reading .sdf .mol2 .pdb for now; 
    # complain if .gninatypes - we're not using any 3D features anyway.
    # if a filename is repeated on contiguous lines and that filename is
    # multi-model, assume the lines correspond to those models and verify we
    # get the expected number.
    # TODO? currently not enforcing contiguity of examples from the same file, 
    # which could lead to a mismatch between the labels and features
    failures = []
    features = []
    count = 0 # track total number of mols seen and compare with expected
    assert method == 'MUV' or method == 'DUD-E', "Only MUV or DUD-E molecular "
    "descriptors supported"

    # how many mols do we expect per filename?
    expected_mols_per_file = pd.DataFrame(mol_list, columns=['file']).groupby('file').size().to_dict()
    # now get the (ordered) list of unique filenames
    unique_files = unique_list(mol_list)

    for molname in unique_files:
        gzipped = False
        base,ext = os.path.splitext(molname)
        fullname = os.path.join(data_root, molname)
        assert ext != '.gninatypes', "Sorry, no gninatypes support currently. "
        "Just pass the starting structure files; if you have multi-model SDFs, "
        "repeat the filename for each example derived from that file and the "
        "script will handle it."
        if ext == '.gz':
            base,ext = os.path.splitext(base)
            assert ext == '.sdf', "Only SDFs can be gzipped for now."
            gzipped = True
        if ext == '.pdb':
            mol = Chem.MolFromPDBFile(fullname)
            mols = [mol]
        elif ext == '.mol2':
            mol = Chem.MolFromMol2File(fullname)
            mols = [mol]
        elif ext == '.sdf':
            if gzipped:
            	mols = Chem.ForwardSDMolSupplier(gzip.open(fullname))
            else:
            	mols = Chem.ForwardSDMolSupplier(fullname)
        # dkoes sometimes typos "ism" for "smi" in filenames...
        elif ext == '.smi' or ext == '.ism':
            with open(fullname, 'r') as f:
                mols = [Chem.MolFromSmiles(line.split()[0]) for line in f]
        else:
            assert 0, 'Unrecognized molecular extension %s' %ext
        for i,mol in enumerate(mols):
            if mol is None:
                print('Problem with molecule %d from file %s' %(i+1,molname))
                failures.append(count)
            else:
                if method == 'DUD-E':
                    features.append(get_dude_descriptors(mol))
                elif method == 'MUV':
                    features.append(get_muv_descriptors(mol))
                else:
                    # shouldn't get here because we already asserted above,
                    # buuuuut just in case
                    print('Unsupported molecular descriptor set %s' %method)
                    sys.exit()
            count += 1
        assert i+1 == expected_mols_per_file[molname], "Got %d mols from %s but expected %d" %(i+1,molname, expected_mols_per_file[molname])

    assert count == len(mol_list), "Saw %d mols but expected %d" %(count, len(mol_list))
    
    print('%d mols successfully parsed, %d failures' %(len(features), len(failures)))
    features = np.asarray(features)
    return FeaturizeOutput(features, failures)

def delete_failure_indices_from_folds(failures, labels, fold_it):
    '''
    Parameters
    ----------
    failures: array_like
        indices of examples to be deleted
    labels: array_like
        example labels
    fold_it: iterable
        iterable yielding fold indices
    
    Returns
    ----------
    labels: array_like
        example labels with failures removed
    fold_it: iterable
        iterable with failures removed
    '''

    # delete labels associated with any mols we failed to parse
    labels = np.delete(labels, failures, axis=0)
 
    # also delete them from the iterator defining the folds...
    # that iterator is actually a bunch of lists of indices;
    # each index will appear in two train fold lists and one test fold, 
    # and we can figure out which
    test_sizes = [len(ttup[1]) for ttup in fold_it]
    start_indices = [part_sum for part_sum in exclusive_scan(size for size in test_sizes)]
    foldnums = list(range(len(test_sizes)))
    index_groups = {}
    for i in foldnums:
        index_groups[i] = []
    # first we'll partition the failed indices by test and train folds
    for failed in failures:
        fold = -1
        for i,start in enumerate(start_indices[1:]):
            if failed < start:
                fold = i
                break
        assert fold != -1, "Failure index %d exceeds data size %d" %(failed, start_indices[-1])
        index_groups[fold].append(failed)

    # the fold we've noted is the test fold; concat the other folds to get the train fold
    new_test_folds = {}
    for fold,failed in index_groups.items():
        # delete from test
        failed = sorted(failed, reverse=True)
        test_failed = [x-start_indices[fold] for x in failed]
        test = fold_it[fold][1]
        for index in test_failed:
            del test[index]
        new_test_folds[fold] = test

    fold_it = []
    for fold in foldnums:
        other_folds = [num for num in foldnums if num != fold]
        train = []
        for num in other_folds:
            train += new_test_folds[num]
        fold_it.append((train,new_test_folds[fold]))

    return FixedOutput(labels, fold_it)

# use sklearn to cross validate, train, and optimize hyperparameters 
def fit_and_cross_validate_model(X_train, y_train, fold_it, param_grid, seed=42, classifier=False):
    '''
    Parameters
    ----------
    X_train: array_like
        training examples
    y_train: array_like
        training labels
    fold_it: iterable
        iterable yielding fold indices
    param_grid: dict
        dictionary mapping hyperparams and values to define the grid search
    seed: int
        random seed to pass to Random Forest model
    classifier: bool
        whether to train a classifier instead of a regression model

    Returns
    ----------
    model: object
        fit model 
    '''

    if classifier:
        rf = RandomForestClassifier(random_state=seed)
        scorer = 'roc_auc'
    else:
        rf = RandomForestRegressor(random_state=seed)
        scorer = 'neg_mean_squared_error'

    grf = GridSearchCV(rf, param_grid, cv=fold_it, scoring=scorer, return_train_score=True)
    grf.fit(X_train, y_train)
    return grf

def plot_cv_results(results, gp1, gp2, classifier=False, figname='gridsearch_results.pdf'):
    '''
    Parameters
    ----------
    results: GridSearchCV results object
    gp1: tuple
        (name, grid parameter to plot on X axis)
    gp2: tuple
        (name, grid parameter to plot as distinct lines)
    classifier: bool
        whether a classifier or regressor model was trained
    figname: str
        name for output figure
    '''

    fig,ax = plt.subplots(figsize=(13, 13))
    plt.title("GridSearchCV Results", fontsize=16)
   
    score_suffix = 'score'  #TODO: this changes for multi-score evaluation
    if classifier:
        plt.ylabel('Score')
    else:
        plt.ylabel('MSE') 

    plt.xlabel(gp1[0])
    
    palette = ['g', 'k', 'goldenrod', 'deepskyblue', 'darkorchid', 'deeppink', 'sienna']
    assert len(gp2[1]) < len(palette), 'Not enough distinct colors in palette'
    this_palette = palette[:len(gp2[1])]
   
    df = pd.DataFrame(results) 
    grouped = df.groupby(['param_%s' %gp1[0], 'param_%s' %gp2[0]], as_index=False)
    plot_df = df.loc[grouped['mean_test_%s' %score_suffix].idxmax()]
    if not classifier:
        for sample in ['train', 'test']:
            plot_df['mean_%s_%s' %(sample, score_suffix)] = plot_df['mean_%s_%s' %(sample, score_suffix)].mul(-1)
    for i,param_val in enumerate(gp2[1]):
        this_plot = plot_df.loc[plot_df['param_%s' %gp2[0]] == param_val]
        color = this_palette[i]
        x = this_plot['param_%s' %gp1[0]].to_numpy(dtype=np.float32)
        for sample, style in (('train', '--'), ('test', '-')):
            sample_score_mean = this_plot['mean_%s_%s' % (sample, score_suffix)].to_numpy(dtype=np.float32)
            sample_score_std = this_plot['std_%s_%s' % (sample, score_suffix)].to_numpy(dtype=np.float32)

            ax.fill_between(x, sample_score_mean - sample_score_std,
                            sample_score_mean + sample_score_std,
                            alpha=0.1 if sample == 'test' else 0, color=color)
            ax.plot(x, sample_score_mean, style, color=color,
                    alpha=1 if sample == 'test' else 0.7,
                    label='%s: %s (%s)' % (gp2[0], param_val, sample))
   
        this_best = this_plot.loc[this_plot['rank_test_%s' % score_suffix].idxmin()]
        best_score = this_best['mean_test_%s' % score_suffix]
        best_x = this_best['param_%s' %gp1[0]]
    
        # Plot a dotted vertical line at the best score for that scorer marked by x
        ax.plot([best_x, ] * 2, [0, best_score],
                linestyle='-.', color=color, marker='x', markeredgewidth=3, ms=8)
    
        # Annotate the best score for that scorer
        ax.annotate('%0.2f' % best_score,
                    (best_x, best_score + 0.005))
    
    plt.legend(loc='best')
    plt.grid(False)
    fig.savefig(figname, bbox_inches='tight')

if __name__=='__main__':
    parser = ArgumentParser(description='Train a Random Forest model for '
            'classification or regression using simple molecular descriptors')
    parser.add_argument('-p', '--prefix', type=str, required=True,
            help="Prefix for training/test files: <prefix>[train|test][num].types")
    parser.add_argument('-n', '--foldnums', type=str, required=False, default=None, 
            help="Fold numbers to run, default is to determine using glob")
    parser.add_argument('-c', '--columns', type=str, default='Label,Affinity,Recfile,Ligfile',
            help='Comma-separated list of column identifiers for folds files, '
            'default is "Label,Affinity,Recfile,Ligfile"')
    parser.add_argument('-r', '--data_root', type=str, default='', 
            help='Common path to join with molnames to generate full location '
            'of files')
    parser.add_argument('-m', '--method', type=str, default='DUD-E',
            help='Descriptor set to use; options are "DUD-E" or "MUV"')
    parser.add_argument('-s', '--seed', type=int, default=42,
            help='Random seed, used for boostrapping and feature sampling')
    parser.add_argument('-f', '--fit_all', action='store_true', 
            help='Fit all numerical columns in types file with RFRegressor. '
            'Default is to use only affinity if available, label if it is not')
    parser.add_argument('-o', '--outprefix', type=str, default='', 
            help='Output prefix for trained random forest pickle and train/test figs')
    args= parser.parse_args()

    fold_data = find_and_parse_folds(args.prefix, args.foldnums, args.columns, args.fit_all)
    print('Generating descriptors...')
    featurize_output = generate_descriptors(fold_data.fnames, args.data_root, args.method)
   
    if featurize_output.failures: 
        print('Removing failed examples')
        out = delete_failure_indices_from_folds(featurize_output.failures, fold_data.labels, fold_data.fold_it)
        labels = out.labels
        fold_it = out.fold_it
    else:
        labels = fold_data.labels
        fold_it = fold_data.fold_it

    colnames = args.columns.split(',')
    if args.fit_all or 'Affinity' in colnames:
        classifier = False
    else:
        classifier = True

    # TODO: maybe too much?
    param_grid = {
        'min_samples_split': [0.1, 0.25, 0.5, 1.0],
        'min_samples_leaf': [0.1, 0.25, 0.5, 1],
        'max_features': ['sqrt', 0.25, 0.5, 0.75, 1.0],
        'n_estimators': [100, 200, 400, 750]
    }
    print('Fitting model')
    if len(labels.shape) == 2 and labels.shape[1] == 1:
        labels = labels.ravel()
    rf = fit_and_cross_validate_model(featurize_output.features, labels, fold_it, param_grid, args.seed, classifier)
    print("best parameters: {}".format(rf.best_params_))
    if not classifier:
        score = -rf.best_score_
    else:
        score = rf.best_score_
    rf_stdev = rf.cv_results_['std_test_score'][rf.best_index_]
    print("best score: {:0.5f} (+/-{:0.5f})".format(score, rf_stdev))

    if not args.outprefix:
        args.outprefix = '%s_RF_%d' %(args.method, os.getpid())
    print('Dumping best model for later')
    dump(rf.best_estimator_, '%s.joblib' %args.outprefix)

    params = list(param_grid.keys())
    gp1 = params[0]
    print('Plotting hyperparameter search results')
    for gp2 in params[1:]:
        plot_cv_results(rf.cv_results_, (gp1,param_grid[gp1]), (gp2,param_grid[gp2]), classifier, 
                        '%s_%s_%s_gridsearch_results.pdf' %(args.outprefix,gp1,gp2))
