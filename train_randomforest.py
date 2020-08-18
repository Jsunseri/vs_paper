#!/bin/env python
import sys,os,re,glob,gzip,math,json
from argparse import ArgumentParser
from collections import namedtuple
from joblib import load,dump

import numpy as np
import pandas as pd
import seaborn as sns
from pandas.api.types import is_numeric_dtype
import matplotlib as mpl
import matplotlib.pyplot as plt

from scipy.stats import pearsonr
from sklearn.metrics import roc_curve, roc_auc_score, make_scorer

from sklearn.linear_model import Lasso

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor

from sklearn.svm import SVC
from sklearn.svm import SVR

from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor

from sklearn.model_selection import cross_validate, cross_val_predict
from sklearn.model_selection import GridSearchCV

from rdkit import Chem
from rdkit.Chem import QED
from rdkit.Chem import rdmolops
from rdkit.Chem import rdMolDescriptors

from openbabel import pybel

from plumbum.cmd import sdsorter,awk

import vspaper_settings

FoldData = namedtuple('FoldData', ['fnames', 'labels', 'fold_it'])
FeaturizeOutput = namedtuple('FeaturizeOutput', ['features', 'failures', 'moltitles'])
FixedOutput = namedtuple('FixedOutput', ['labels', 'fold_it'])
FitOutput = namedtuple('FitOutput', ['test_scores', 'y_pred', 'y_true'])

classifiers = (KNeighborsClassifier, DecisionTreeClassifier, 
               RandomForestClassifier, SVC, GradientBoostingClassifier)

regressors = (Lasso, KNeighborsRegressor, DecisionTreeRegressor, 
               RandomForestRegressor, SVR, GradientBoostingRegressor)

no_init_params = (Lasso, KNeighborsRegressor, KNeighborsClassifier, SVR, SVC)

seedonly = (GradientBoostingRegressor, DecisionTreeRegressor, GradientBoostingClassifier, 
            DecisionTreeClassifier)

methodnames = {KNeighborsClassifier: 'KNN', SVC: 'SVM', GradientBoostingClassifier: 'GBT', 
               DecisionTreeClassifier: 'DT', RandomForestClassifier: 'RF', 
               Lasso: 'Lasso', KNeighborsRegressor: 'KNN', SVR: 'SVM', 
               GradientBoostingRegressor: 'GBT', DecisionTreeRegressor: 'DT', 
               RandomForestRegressor: 'RF'
              }

# TODO: maybe too much? min_samples_split at least seems to just be the
# best at 0.1, no real need to sample
param_grids = {'RF': 
              {'min_samples_split': [0.1, 0.25, 0.5, 1.0],
              'min_samples_leaf': [0.1, 0.25, 0.5, 1],
              'max_features': ['sqrt', 0.25, 0.5, 0.75, 1.0],
              'n_estimators': [100, 200, 400, 750]
              }, 
               'KNN':
              {'n_neighbors': list(range(1,30)), 
               'p': [1,2]
              },
               'SVM':
              {'C': [0.25, 0.5, 0.75, 1.0],
               'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
               'gamma': ['scale', 'auto'],
               'epsilon': [0.025, 0.1, 0.2]
              },
               'GBT':
              {'n_estimators': [100, 200, 400, 750],
               'loss': ['ls', 'lad', 'huber', 'quantile'],
               'learning_rate': [0.01, 0.1, 0.25],
               'subsample': [0.1, 0.25, 0.5, 0.75, 1.0], 
               'min_samples_split': [0.1, 0.25, 0.5, 1.0, 2, 5],
               'min_samples_leaf': [0.1, 0.25, 0.5, 1],
               'max_depth': list(range(1,15)), 
               'max_features': ['auto', 'sqrt', 'log2', None], 
               'alpha': [0.8, 0.9, 0.95]
              },
               'DT':
              {'min_samples_leaf': [0.1, 0.25, 0.5, 1],
               'criterion': ['mse', 'friedman_mse', 'mae'],
               'max_depth': [None, 3, 10],
               'max_features': ['auto', 'sqrt', 'log2', None]
              },
               'Lasso':
              {'alpha': [0.5, 0.75, 1.0],
               'normalize': [True, False], 
              }
}

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

def find_and_parse_folds(prefix, foldnums='', columns='Label,Affinity,Recfile,Ligfile', use_all=False):
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
    use_all: bool, optional
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
    # if we don't find explicit train/test folds, check for <prefix>.types and
    # if so we leave fold_it empty and use the default sklearn CV behavior
    files = []
    if not foldnums:
        foldnums = set()
        glob_files = glob.glob(prefix + '*')
        pattern = r'(%s)(test)?(\d+)?\.types$' % (prefix)
        for file in glob_files:
            match = re.match(pattern, file)
            if match:
                if match.group(3) is not None:
                    foldnums.add(int(match.group(3)))
                else:
                    files.append(file)
        foldnums = list(foldnums)
    else:
        foldnums = [int(i) for i in foldnums.split(',') if i]
    foldnums.sort()
    for i in foldnums:
        fname = '%stest%d.types' % (prefix, i)
        assert os.path.isfile(fname), '%s file not found' %fname
        files.append(fname)

    # we *require* a ligfile and at least one target for
    # regression/classification. by default we look for label and affinity; if
    # only a label is present we use it for classification, and if both are
    # present we use just the affinity for regression. sklearn doesn't seem to
    # allow mixed-task multi-task models, so if --use_all is passed, we train a
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
    print('Got %d examples from %d files\n' %(total_examples, len(files)))

    def get_indices(numlist, idx):
        '''
        return (train,test) indices, where numlist is the number of examples
        associated with each test fold, idx is the test fold for this round and
        the other test folds will be concatenated to constitute the train set
        '''
        train = []
        start_indices = [part_sum for part_sum in exclusive_scan(size for size in numlist)]
        for i,start in enumerate(start_indices[:-1]):
            if i != idx:
                train += list(range(start, start+numlist[i]))
            else:
                test = list(range(start, start+numlist[i]))
        return (train,test)

    if foldnums:
        fold_it = [get_indices(elems_per_df, idx) for idx in range(len(elems_per_df))]
    else:
        fold_it = []

    # get lig filenames
    df = pd.concat(df_list, ignore_index=True, sort=False)
    fnames = df['Ligfile'].to_numpy()

    # get y_values
    allcols = df.columns
    ycols = []
    if use_all:
        for col in allcols:
            if is_numeric_dtype(df[col]):
                ycols.append(col)
    else:
        if 'Affinity' in allcols:
            ycols = ['Affinity']
        elif 'Label' in allcols:
            ycols = ['Label']
        else:
            assert 0, 'Custom fit targets not implemented yet. Pass --use_all '
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

    if 5 in atom_counts:
        muv_descriptors.append(atom_counts[5])  # boron
    else:
        muv_descriptors.append(0)
    if 35 in atom_counts:
        muv_descriptors.append(atom_counts[35]) # bromine
    else:
        muv_descriptors.append(0)
    if 6 in atom_counts:
        muv_descriptors.append(atom_counts[6])  # carbon
    else:
        muv_descriptors.append(0)
    if 17 in atom_counts:
        muv_descriptors.append(atom_counts[17]) # chlorine
    else:
        muv_descriptors.append(0)
    if 9 in atom_counts:
        muv_descriptors.append(atom_counts[9])  # fluorine
    else:
        muv_descriptors.append(0)
    if 53 in atom_counts:
        muv_descriptors.append(atom_counts[53]) # iodine
    else:
        muv_descriptors.append(0)
    if 7 in atom_counts:
        muv_descriptors.append(atom_counts[7])  # nitrogen
    else:
        muv_descriptors.append(0)
    if 8 in atom_counts:
        muv_descriptors.append(atom_counts[8])  # oxygen
    else:
        muv_descriptors.append(0)
    if 15 in atom_counts:
        muv_descriptors.append(atom_counts[15]) # phosphorus
    else:
        muv_descriptors.append(0)
    if 16 in atom_counts:
        muv_descriptors.append(atom_counts[16]) # sulfur
    else:
        muv_descriptors.append(0)
    
    # TODO: will this actually get them all?
    chiral_centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
    muv_descriptors.append(len(chiral_centers))
    muv_descriptors.append(rdMolDescriptors.CalcNumRings(mol))

    return muv_descriptors

# we generate descriptors with rdkit and give the user the option of using the
# DUD-E or MUV descriptors
def generate_descriptors(mol_list, data_root=[], method='DUD-E', use_babel=False, extra_descriptors=None):
    '''
    Parameters
    ----------
    mol_list: array_like
        list of molecule filenames
    data_root: array_like
        common path to join with molnames to generate full location, provided
        as a list which are tried in order until file is found or the list is
        exhausted
    method: {'DUD-E', 'MUV'}, optional
        Which set of descriptors to use; either DUD-E (the default) or MUV
    use_babel: bool
        parse files and generate descriptors with babel instead of rdkit
    extra_descriptors: array_like, optional
        N_mols X N_descs array of additional user-provided descriptors

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
    # which could lead to a mismatch between the labels and features if the 
    # user doesn't respect the convention
    failures = []
    features = []
    moltitles = []
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
        if data_root:
            found = False
            for path in data_root:
                fullname = os.path.join(path, molname)
                if os.path.isfile(fullname):
                    found = True
                    break
            assert found, '%s does not exist in any user-provided directories.' %molname
        else:
            fullname = molname
        if use_babel:
            for i,mol in enumerate(pybel.readfile(ext.split('.')[-1], fullname)):
                if mol is not None:
                    desc = mol.calcdesc()
                    desc['charge'] = mol.charge
                    if extra_descriptors is not None:
                        features.append([desc[d] for d in ['MW','HBA1','HBD','rotors','logP','charge']] + 
                                        extra_descriptors[count,:].tolist())
                    else:
                        features.append([desc[d] for d in ['MW','HBA1','HBD','rotors','logP','charge']])
                else:
                    print('Problem with molecule %d from file %s\n' %(i+1,molname))
                    failures.append(count)
                count += 1
            assert i+1 == expected_mols_per_file[molname], "Got %d mols from %s but expected %d" %(i+1,molname, expected_mols_per_file[molname])
        else:
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
              # detour here to get the molnames with sdsorter; if we don't have
              # SDFs or SMIs...sorry, you're on your own for names
              moltitles += (sdsorter['-print', '-omit-header', fullname] | awk['{print $2}'])().strip().split('\n')
          # dkoes sometimes typos "ism" for "smi" in filenames...
          elif ext == '.smi' or ext == '.ism':
              with open(fullname, 'r') as f:
                  mols = [Chem.MolFromSmiles(line.split()[0]) for line in f]
                  f.seek(0)
                  moltitles += [line.strip().split()[-1] for line in f]
          else:
              assert 0, 'Unrecognized molecular extension %s' %ext
          for i,mol in enumerate(mols):
              if mol is None:
                  print('Problem with molecule %d from file %s\n' %(i+1,molname))
                  failures.append(count)
              else:
                  if method == 'DUD-E':
                      if extra_descriptors is not None:
                          features.append(get_dude_descriptors(mol) +
                                  extra_descriptors[count,:].tolist())
                      else:
                          features.append(get_dude_descriptors(mol))
                  elif method == 'MUV':
                      if extra_descriptors is not None:
                          features.append(get_muv_descriptors(mol) +
                                  extra_descriptors[count,:].tolist())
                      else:
                          features.append(get_muv_descriptors(mol))
                  else:
                      # shouldn't get here because we already asserted above,
                      # buuuuut just in case
                      print('Unsupported molecular descriptor set %s\n' %method)
                      sys.exit()
              count += 1
          assert i+1 == expected_mols_per_file[molname], "Got %d mols from %s but expected %d" %(i+1,molname, expected_mols_per_file[molname])

    assert count == len(mol_list), "Saw %d mols but expected %d" %(count, len(mol_list))
    
    print('%d mols successfully parsed, %d failures\n' %(len(features), len(failures)))

    features = np.asarray(features)
    return FeaturizeOutput(features, failures, moltitles)

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
 
    # also delete them from the iterable defining the folds, if it is populated
    # that iterable is actually a bunch of lists of indices;
    # each index will appear in two train fold lists and one test fold, 
    # and we can figure out which
    if fold_it:
        test_sizes = [len(ttup[1]) for ttup in fold_it]
        start_indices = [part_sum for part_sum in exclusive_scan(size for size in test_sizes)]
        foldnums = list(range(len(test_sizes)))
        index_groups = {}
        failed_counts = [0]*len(foldnums)
        # the failures were removed from X/y, so we just need to 
        # figure out how many failures map to each fold and 
        # slice the list to chop off the cumulative number of failures
        for failed in failures:
            fold = -1
            for i,start in enumerate(start_indices[1:]):
                if failed < start:
                    fold = i
                    break
            assert fold != -1, "Failure index %d exceeds data size %d" %(failed, start_indices[-1])
            failed_counts[fold] += 1

        cumulative_failed_counts = [part_sum for part_sum in exclusive_scan(size for size in failed_counts)]
        # the fold we've noted is the test fold; concat the other folds to get the train fold
        new_test_folds = {}
        for fold,failed in enumerate(cumulative_failed_counts[1:]):
            newsize = test_sizes[fold] - failed
            test = sorted(fold_it[fold][1])
            new_test_folds[fold] = test[:newsize]

        fold_it = []
        for fold in foldnums:
            other_folds = [num for num in foldnums if num != fold]
            train = []
            for num in other_folds:
                train += new_test_folds[num]
            fold_it.append((train,new_test_folds[fold]))

    return FixedOutput(labels, fold_it)

def fit_model(X_train, y_train, params=None, njobs=1, seed=42, classifier=False):
    '''
    Parameters
    ----------
    X_train: array_like
        training examples
    y_train: array_like
        training labels
    params: dict
        dict of hyperparameters for random forest
    njobs: int
        number of sklearn jobs, corresponding to parallel processes during CV
    seed: int
        random seed to pass to Random Forest model
    classifier: bool
        whether to train a classifier instead of a regression model

    Returns
    ----------
    model: object
        fit model 
    '''
    if params is None:
        params = {'min_samples_split': 0.1,
                'n_estimators': 200, 'min_samples_leaf': 1, 'max_features': 0.5}
    if classifier:
        rf = RandomForestClassifier(random_state=seed, 
                n_jobs=njobs, class_weight='balanced_subsample', **params)
    else:
        rf = RandomForestRegressor(random_state=seed, n_jobs=njobs, **params)
    rf.fit(X_train, y_train)
    return rf

def plot_classifier(true, pred, method, fig, grid_length, grid_width, color):
    '''
    Parameters
    ----------
    true: array_like
        true data values
    pred: array_like
        predicted values
    method: str
        name of method
    fig: matplotlib::Figure
        figure object to plot on
    grid_length: int
        length of plot grid
    grid_width: int
        width of plot grid
    color: matplotlib color
        anything matplotlib can interpret as a color
    '''
    sub_ax = plt.subplot2grid((grid_length,grid_width),
            (plot_num // grid_width, plot_num % grid_width),
            fig=fig)
    sub_ax.set_aspect('equal')
    fpr,tpr,_ = roc_curve(true, pred)
    subax.plot(fpr, tpr, color=color,
            label=method, lw=5, zorder=2) 
    auc, _ = roc_auc_score(true, pred)
    sub_ax.annotate(r'AUC = {0:.2}$'.format(auc), xy=(.1, .9),
            xycoords=g.transAxes, bbox=props)
    sub_ax.set(ylim=(0.0, 1.0))
    sub_ax.set(xlim=(0.0, 1.0))
    sub_ax.plot([0, 1], [0, 1], color='gray', lw=5, linestyle='--', zorder=1)
    sub_ax.title(method)

def plot_regressor(true, pred, method, color):
    '''
    Parameters
    ----------
    true: array_like
        true data values
    pred: array_like
        predicted values
    method: str
        name of method
    color: matplotlib color
        anything matplotlib can interpret as a color
    '''
    g = sns.jointplot(true, pred, color=color, alpha=0.7)
    r, _ = pearsonr(true, pred)
    g.annotate(pearsonr)
    g.set_axis_labels("-logK", "Prediction")
    g.fig.suptitle(method)
    g.savefig('%s_regressor_fit.pdf' %method)

def do_fit(estimator, X_train, y_train, fold_it=5):
    '''
    Parameters
    ----------
    estimator: estimator object implementing ‘fit’ and ‘predict’
        the object to use to fit the data
    X_train: array_like
        training examples
    y_train: array_like
        training labels
    fold_it: iterable or int
        iterable yielding fold indices or int K for K-fold 

    Returns
    ----------
    test_scores: array_like
        CV score results
    y_pred: array_like
        cumulative predictions for cross validated test examples
    y_true: array_like
        true values corresponding to predictions
    '''
    def pearsonfunc(y_true, y_pred):
        return pearsonr(y_true, y_pred)[0]
    pearsonscore = make_scorer(pearsonfunc)

    if issubclass(estimator, classifiers):
        scoring = 'roc_auc'
        classifier = True
    else:
        scoring = pearsonscore
        classifier = False

    cv_results = cross_validate(estimator, X_train, y_train, cv=fold_it,
            scoring=scoring)

    if isinstance(fold_it, int):
        if classifier:
            y_pred = cross_val_predict(estimator, X_train, y_train, cv=fold_it, method='predict_proba')[:,-1]
        else:
            y_pred = cross_val_predict(estimator, X_train, y_train, cv=fold_it)[:,-1]
        y_true = y_train
    else:
        y_pred = []
        y_true = []
        for (train_indices,test_indices) in fold_it:
            this_xtrain = X_train.take(train_indices,axis=0)
            this_ytrain = y_train.take(train_indices,axis=0)
            this_xtest = X_train.take(test_indices,axis=0)
            estimator.fit(this_xtrain, this_ytrain)
            if classifier:
                y_pred += estimator.predict_proba(this_xtest).tolist()
            else:
                y_pred += estimator.predict(this_xtest).tolist()
            y_true += y_train.take(test_indices,axis=0).tolist()
        y_pred = np.array(y_pred)
        y_true = np.array(y_true)
    return FitOutput(cv_results['test_score'], y_pred, y_true)

def fit_all_models(X_train, y_train, fold_it=[], njobs=1, seed=42, classifier=False):
    '''
    Parameters
    ----------
    X_train: array_like
        training examples
    y_train: array_like
        training labels
    fold_it: iterable
        iterable yielding fold indices
    njobs: int
        number of sklearn jobs, corresponding to parallel processes during CV
    seed: int
        random seed to pass to Random Forest model
    classifier: bool
        whether to train a classifier instead of a regression model

    Returns
    ----------
    '''
    # for now, don't dump any of the fits, just plot results 
    if not fold_it: fold_it=5

    data = {}
    data['Method'] = []
    palette = sns.color_palette("hls", n_colors=8, desat=.5).as_hex()
    methodcolors = {}
    box_fig,box_ax = plt.subplots()
    # plot grid of ROC curves for fit and boxplot of AUCs for CV
    if classifier:
        data['AUC'] = []
        total_plots = len(classifiers)
        grid_width = int(math.ceil(math.sqrt(total_plots)))
        grid_length = int(math.ceil(float(total_plots)/grid_width))
        fig,ax = plt.subplots(figsize=(16,16))
        for plot_num,model in enumerate(classifiers):
            color = palette[plot_num]
            methodname = methodnames[model]
            methodcolors[methodname] = color
            if issubclass(model, RandomForestClassifier):
                estimator = model(random_state=seed, n_jobs=njobs, class_weight='balanced_subsample')
            elif issubclass(model, no_init_params):
                estimator = model()
            elif issubclass(model, seedonly):
                estimator = model(random_state=seed)
            else:
                estimator = model(n_jobs=njobs, random_state=seed)
            fit_output = do_fit(estimator, X_train, y_train, fold_it)
            plot_classifier(y_train, preds, methodname, fig, grid_length,
                    grid_width, color)
            vals = fit_output.test_scores
            for val in vals:
                data['Method'].append(methodname)
                data['AUC'].append(val)

        fig.savefig('CV_classifier_fit_severalmodels.pdf')
        df = pd.DataFrame(data)
        sns.stripplot(x='Method', y='AUC',
                data=df, 
                split=True, edgecolor='black', size=10, linewidth=0,
                linewidths=0.5, jitter = True,
                palette=methodcolors, marker='o',
                ax=box_ax)
        sns.boxplot(x='Method', y='AUC', data=df, 
                color='white', ax=box_ax)
        box_fig.savefig('auc_boxplot_severalmodels.pdf')
    # plot jointplots of preds vs actual, and boxplots of the fold R values
    else:
        data['R'] = []
        summarystr = ''
        methodcolors = {}

        for plot_num,model in enumerate(regressors):
            color = palette[plot_num]
            methodname = methodnames[model]
            methodcolors[methodname] = color
            if issubclass(model, no_init_params):
                estimator = model()
            elif issubclass(model, seedonly):
                estimator = model(random_state=seed)
            else:
                estimator = model(n_jobs=njobs, random_state=seed)
            fit_output = do_fit(estimator, X_train, y_train, fold_it)
            true = fit_output.y_true
            preds = fit_output.y_pred
            plot_regressor(true, preds, methodname, color)
            R = pearsonr(true, preds)[0]
            rmse = np.sqrt(np.mean(np.square(true - preds)))
            summarystr += '\n%s R=%0.3f, RMSE=%0.3f' %(methodname, R, rmse)
            vals = fit_output.test_scores
            for val in vals:
                data['Method'].append(methodname)
                data['R'].append(val)

   # now for the boxplots with dots for each fold
        df = pd.DataFrame(data)
        sns.stripplot(x='Method', y='R',
                data=df, 
                split=True, edgecolor='black', size=10, linewidth=0,
                linewidths=0.5, jitter = True,
                palette=methodcolors, marker='o',
                ax=box_ax)
        sns.boxplot(x='Method', y='R', data=df, 
                color='white', ax=box_ax)
        box_fig.savefig('pearsonr_boxplot_severalmodels.pdf')
        print(summarystr)
    return 

# use sklearn to cross validate, train, and optimize hyperparameters 
def fit_and_cross_validate_model(estimator, X_train, y_train, param_grid, scoring, fold_it, n_jobs=-1):
    '''
    Parameters
    ----------
    estimator: estimator object implementing ‘fit’ and ‘predict’
        the object to use to fit the data
    X_train: array_like
        training examples
    y_train: array_like
        training labels
    param_grid: dict
        dictionary mapping hyperparams and values to define the grid search
    scoring: str or array_like of str
        how to score gridsearch outputs; if array_like, first will be used for refit
    fold_it: iterable
        iterable yielding fold indices; if None, just use the default CV
        strategy (currently 5-fold)

    Returns
    ----------
    results: object
        output of GridSearchCV
    '''

    if not isinstance(scoring, str):
        refit = scoring[0]
    else:
        refit = scoring
    if fold_it:
        out = GridSearchCV(estimator, param_grid, cv=fold_it, scoring=scoring, refit=refit, 
                           n_jobs=n_jobs, return_train_score=True)
    else:
        out = GridSearchCV(estimator, param_grid, scoring=scoring, refit=refit, 
                           n_jobs=n_jobs, return_train_score=True)
    out.fit(X_train, y_train)
    return out

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

    mpl.rc('text', usetex=True)
    fig,ax = plt.subplots(figsize=(13, 13))
    plt.title("GridSearchCV Results", fontsize=16)
   
    if classifier:
        plt.ylabel('Score')
        score_suffix = 'score'
    else:
        plt.ylabel(r'$R^{2}$') 
        score_suffix = 'r2'

    plt.xlabel(gp1[0].replace('_', '\_'))
   
    palette = ['g', 'k', 'goldenrod', 'deepskyblue', 'darkorchid', 'deeppink', 'sienna']
    ncolors = len(gp2[1])
    if ncolors > len(palette):
        cmap = matplotlib.cm.get_cmap('Set3')
        palette = [cmap[val] for val in np.linspace(0, 1, ncolors)]
    
    df = pd.DataFrame(results) 
    grouped = df.replace({'param_%s' %gp2[0]: {None: -1}}).groupby(['param_%s' %gp1[0], 'param_%s' %gp2[0]], as_index=False)
    plot_df = df.loc[grouped['mean_test_%s' %score_suffix].idxmax()].replace({'param_%s' %gp2[0]: {-1: None}})
    if not classifier and score_suffix == 'score' or score_suffix == 'neg_mean_squared_error':
        for sample in ['train', 'test']:
            plot_df['mean_%s_%s' %(sample, score_suffix)] = plot_df['mean_%s_%s' %(sample, score_suffix)].mul(-1)
    for i,param_val in enumerate(gp2[1]):
        if param_val is None:
            this_plot = plot_df.loc[plot_df['param_%s' %gp2[0]].isnull()]
        else:
            this_plot = plot_df.loc[plot_df['param_%s' %gp2[0]] == param_val]
        color = palette[i]
        x = this_plot['param_%s' %gp1[0]].to_numpy(dtype=np.float32)
        for sample, style in (('train', '--'), ('test', '-')):
            sample_score_mean = this_plot['mean_%s_%s' % (sample, score_suffix)].to_numpy(dtype=np.float32)
            sample_score_std = this_plot['std_%s_%s' % (sample, score_suffix)].to_numpy(dtype=np.float32)

            ax.fill_between(x, sample_score_mean - sample_score_std,
                            sample_score_mean + sample_score_std,
                            alpha=0.1 if sample == 'test' else 0, color=color)
            ax.plot(x, sample_score_mean, style, color=color,
                    alpha=1 if sample == 'test' else 0.7,
                    label='%s: %s (%s)' % (gp2[0].replace('_','\_'), str(param_val).replace('_','\_'), sample))
  
        this_best = this_plot.loc[this_plot['rank_test_%s' % score_suffix].idxmin()]
        best_score = this_best['mean_test_%s' % score_suffix]
        best_x = this_best['param_%s' %gp1[0]]
    
        # Plot a dotted vertical line at the best score for that scorer marked by x
        ax.plot([best_x, ] * 2, [0, best_score],
                linestyle='-.', color=color, marker='x', markeredgewidth=3, ms=8)
    
        # Annotate the best score for that scorer
        ax.annotate('%0.2f' % best_score,
                    (best_x, best_score + 0.005))
   
    ylims = ax.get_ylim() 
    ax.set_ylim(0, ylims[1])
    plt.legend(loc='best')
    plt.grid(False)
    fig.savefig(figname, bbox_inches='tight')

if __name__=='__main__':
    parser = ArgumentParser(description='Train a Random Forest model for '
            'classification or regression using simple molecular descriptors')
    parser.add_argument('-p', '--prefix', type=str, default='',
            help='Prefix for training/test files: '
            '<prefix>[train|test][num].types will use those partitions for cross '
            'validation, while a single file <prefix>.types will do 5-fold '
            'randomly partitioned cross validation')
    parser.add_argument('-n', '--foldnums', type=str, required=False, default=None, 
            help="Fold numbers to run, default is to determine using glob")
    parser.add_argument('-c', '--columns', type=str, default='Label,Affinity,Recfile,Ligfile',
            help='Comma-separated list of column identifiers for folds files, '
            'default is "Label,Affinity,Recfile,Ligfile"')
    parser.add_argument('-d', '--descriptor_file', type=str, default='',
            help='Provide precomputed descriptors for the mols')
    parser.add_argument('-e', '--extra_descriptors', type=str, default='',
            help='Provide an additional file with precomputed descriptors')
    parser.add_argument('-r', '--data_root', nargs='*', default=[], 
            help='Common path to join with molnames to generate full location '
            'of files; can pass multiple, which will be tried in order')
    parser.add_argument('-m', '--method', type=str, default='DUD-E',
            help='Descriptor set to use; options are "DUD-E" or "MUV"')
    parser.add_argument('-s', '--seed', type=int, default=42,
            help='Random seed, used for boostrapping and feature sampling')
    parser.add_argument('-u', '--use_all', action='store_true', 
            help='Use all numerical columns in types file with RFRegressor. '
            'Default is to use only affinity if available, label if it is not')
    parser.add_argument('-f', '--fit_all', action='store_true', 
            help='Fit all available models with default parameters: Lasso, KNN, '
            'SVM, RF, GBT, DT')
    parser.add_argument('-j', '--just_fit', action='store_true',
            help='Just fit using reasonable settings, no hyperparameter sampling')
    parser.add_argument('-nc', '--ncpus', type=int, default=1, 
            help='Number of processes to launch for model fitting; default=1')
    parser.add_argument('-b', '--use_babel', action='store_true', 
            help='Use OpenBabel to read in molecules and generate descriptors (default is to use rdkit)')
    parser.add_argument('-o', '--outprefix', type=str, default='', 
            help='Output prefix for trained random forest pickle and train/test figs')
    parser.add_argument('--dump', action='store_true',
            help='Dump best model after fit/optimization.')
    parser.add_argument('--hyperparms', type=str, default='',
            help='Provide input json file specifying hyperparameter values for specific models;'
            'must specify "modelname":str and "params":dict')
    args = parser.parse_args()

    # for now
    if args.method == 'MUV':
        assert not args.use_babel, 'For now, featurization with babel only supports DUD-E descriptors'

    # include features from user-provided file of precomputed features, if available
    if args.extra_descriptors:
        extra_df = pd.read_csv(args.extra_descriptors, delim_whitespace=True, header=None)
        extra_descs = extra_df.to_numpy()
    else:
        extra_descs = None

    # read in precomputed descriptors to save time if provided (hopefully this
    # saves time)
    if args.descriptor_file:
        print('Reading in precomputed descriptors; additional descriptors/mols '
                'will not be included, but extra descriptors will be '
                'concatenated\n')
        fold_data,featurize_output = load(args.descriptor_file)
        if args.extra_descriptors:
            shape = featurize_output.features.shape
            extra_shape = extra_descs.shape
            assert shape[0] == extra_shape[0], 'First dim of descriptors and '
            'extra_descriptors must match, but they are %d and %d' %(shape[0], extra_shape[0])
            featurize_output.features = np.append(featurize_output.features,
                    extra_descs, axis=1)
    else:
        assert args.prefix, 'Need fold files if precomputed descriptors are not provided'
        fold_data = find_and_parse_folds(args.prefix, args.foldnums, args.columns, args.use_all)
        if args.extra_descriptors:
            assert extra_df.shape[0] == len(fold_data.labels), 'Extra descriptor file should have the same number of examples as the folds but has %s instead of %s' %(extra_df.shape[0], len(fold_data.labels))
        # TODO: multiprocess this? would need to rewrite how fold iteratable is made
        print('Generating descriptors...\n')
        featurize_output = generate_descriptors(fold_data.fnames, args.data_root,
                args.method, args.use_babel, extra_descs)
   
    if featurize_output.failures: 
        print('Removing failed examples\n')
        out = delete_failure_indices_from_folds(featurize_output.failures, fold_data.labels, fold_data.fold_it)
        labels = out.labels
        fold_it = out.fold_it
    else:
        labels = fold_data.labels
        fold_it = fold_data.fold_it

    colnames = args.columns.split(',')
    if args.use_all or 'Affinity' in colnames:
        classifier = False
    else:
        classifier = True

    # TODO: reshape instead? what happens with more than one target value?
    if len(labels.shape) == 2 and labels.shape[1] == 1:
        labels = labels.ravel()

    if not args.descriptor_file:
        print('Dumping computed descriptors for reuse.\n')
        dump((FoldData([], labels, fold_it), FeaturizeOutput(featurize_output.features, [], [])), 
                '%sdescriptors.joblib' %args.outprefix)

    if args.hyperparms:
        hyperparms = json.load(open(args.hyperparms))
        available_methods = [elem['modelname'] for elem in hyperparms]
        params = [elem['params'] for elem in hyperparms]
        for i,mname in enumerate(available_methods):
            paramdicts[mname] = params[i]
    # TODO: clean up these options, things changed at the end of last week
    if args.just_fit:
        if classifier:
            for model in classifiers:
                methodname = methodnames[model]
                print('Since --just_fit was passed, doing a single fit of $s with params %s' %str(methodname,params))
                params = paramdict[mname] if mname in params else {}
                if issubclass(model, RandomForestClassifier):
                    estimator = model(random_state=args.seed, class_weight='balanced_subsample', **params)
                elif issubclass(model, no_init_params):
                    estimator = model(**params)
                else:
                    estimator = model(random_state=args.seed, **params)
                outm = estimator.fit(featurize_output.features, labels)
                print("{} mean accuracy: {:0.5f}".format(methodname, outm.score(featurize_output.features, labels)))
                if args.dump:
                    print('Dumping fit model for later\n')
                    dump(outm, '%s_%s.joblib' %(modelname, args.outprefix))
            for model in regressors:
                methodname = methodnames[model]
                print('Since --just_fit was passed, doing a single fit of $s with params %s' %str(methodname,params))
                params = paramdict[mname] if mname in params else {}
                if issubclass(model, no_init_params):
                    estimator = model(**params)
                else:
                    estimator = model(random_state=args.seed, **params)
                outm = estimator.fit(featurize_output.features, labels)
                print("{} R^2: {:0.5f}".format(methodname, outm.score(featurize_output.features, labels)))
                if args.dump:
                    print('Dumping fit model for later\n')
                    dump(outm, '%s_%s.joblib' %(modelname, args.outprefix))
    elif args.fit_all:
        fit_all_models(featurize_output.features, labels, fold_it, args.ncpus, args.seed, classifier)
    else:
        print('Performing hyperparameter grid search for all models\n')
        if classifier:
            for model in classifiers:
                methodname = methodnames[model]
                scoring = 'roc_auc'
                if issubclass(model, RandomForestClassifier):
                    estimator = model(random_state=args.seed, class_weight='balanced_subsample')
                elif issubclass(model, no_init_params):
                    estimator = model()
                else:
                    estimator = model(random_state=args.seed)
                param_grid = param_grids[methodname]
                gridsearch_out = fit_and_cross_validate_model(estimator, featurize_output.features, labels, 
                     param_grid, scoring, fold_it, args.ncpus)
                print("{} best parameters: {}".format(methodname, gridsearch_out.best_params_))
                score = gridsearch_out.best_score_
                out_stdev = gridsearch_out.cv_results_['std_test_score'][gridsearch_out.best_index_]

                print("best {}: {:0.5f} (+/-{:0.5f})".format('score', score, out_stdev))

                params = list(param_grid.keys())
                gp1 = params[0]
                print('Plotting hyperparameter search results\n')
                for gp2 in params[1:]:
                    plot_cv_results(gridsearch_out.cv_results_, (gp1,param_grid[gp1]), (gp2,param_grid[gp2]), classifier, 
                                    '%s_%s_%s_%s_gridsearch_results.pdf' %(args.outprefix,methodname,gp1,gp2))
        else:
            for model in regressors:
                methodname = methodnames[model]
                scoring = ['r2', 'neg_mean_squared_error']
                if issubclass(model, no_init_params):
                    estimator = model()
                else:
                    estimator = model(random_state=args.seed)
                param_grid = param_grids[methodname]
                gridsearch_out = fit_and_cross_validate_model(estimator, featurize_output.features, labels, 
                     param_grid, scoring, fold_it, args.ncpus)
                print("{} best parameters: {}".format(methodname, gridsearch_out.best_params_))
                r2 = gridsearch_out.cv_results_['mean_test_r2'][gridsearch_out.best_index_]
                r2_stdev = gridsearch_out.cv_results_['std_test_r2'][gridsearch_out.best_index_]
                print("best R2: {:0.5f} (+/-{:0.5f})".format(r2, r2_stdev))
                score = -gridsearch_out.cv_results_['mean_test_neg_mean_squared_error'][gridsearch_out.best_index_]
                out_stdev = gridsearch_out.cv_results_['std_test_neg_mean_squared_error'][gridsearch_out.best_index_]
                print("best {}: {:0.5f} (+/-{:0.5f})".format('MSE', score, out_stdev))

                params = list(param_grid.keys())
                gp1 = params[0]
                print('Plotting hyperparameter search results\n')
                for gp2 in params[1:]:
                    plot_cv_results(gridsearch_out.cv_results_, (gp1,param_grid[gp1]), (gp2,param_grid[gp2]), classifier, 
                                    '%s_%s_%s_%s_gridsearch_results.pdf' %(args.outprefix,methodname,gp1,gp2))

        if args.dump:
            if not args.outprefix:
                args.outprefix = '%s_RF_%d' %(args.method, os.getpid())
            print('Dumping best model for later\n')
            dump(gridsearch_out.best_estimator_, '%s.joblib' %args.outprefix)

