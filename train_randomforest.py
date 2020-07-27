#!/bin/env python
import rdkit.AllChem as Chem
import numpy as np
import sys,os,glob
import pandas as pd
from argparse import ArgumentParser
from rdkit import rdMolDescriptors
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from pandas.api.types import is_numeric_dtype
from joblib import dump

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
    assert 'Label' or 'Affinity' in column_names, "Currently a 'Label' or "
        "'Affinity' column are required."
    assert 'Ligfile' in column_names, "Path to ligand file required."

    # parse files with pandas; since we have allowed the column layout to vary,
    # we default to the "normal" version but allow the user to specify others.
    df_list = [pd.read_csv(fname, names=column_names, delim_whitespace=True) for fname in files.values()]

    # fill in list of tuples arranged like [(train0, test0), (train1, test1), ..., (trainN, testN)]
    # where the elements are arrays of indices associated with these folds
    elems_per_df = [df.shape[0] for df in df_list]
    total_examples = sum(elems_per_df)

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
                train += range(start, start+num)
            else:
                test = range(start, start+num)
        return (train,test)

    fold_it = [get_indices(elems_per_df, idx) for idx in range(len(elems_per_df))]

    # get lig filenames
    df = pd.concat(df_list, ignroe_index=True, sort=False)
    fnames = df['Ligfile'].to_numpy()

    # get y_values
    ycols = []
    for col in df.columns:
        if is_numeric_dtype(df[col]):
            possible_yvals.append(col)
    if not fit_all:
        if 'Affinity' in ycols:
            ycols = ['Affinity']
        elif 'Label' in ycols:
            ycols = ['Label']
        else:
            assert 0, "Custom fit targets not implemented yet. Pass --fit_all '
            'or specify target column as 'Affinity'"
    labels = df[ycols].to_numpy()
    return({'fnames':fnames, 'labels': labels, 'fold_it':fold_it})

def dude_descriptors(mol):
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

def muv_descriptors(mol):
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

# we generate descriptors with rdkit and give the user the option of using the
# DUD-E or MUV descriptors
def generate_descriptors(mol_list, num_mols=0, data_root='', method='DUD-E'):
    '''
    Parameters
    ----------
    mol_list: array_like
        list of molecule filenames
    num_mols: int
        number of total mols expected
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
    # complain if .gninatypes - we're not using any 3D features anyway
    count = 0
    failures = []
    assert method == 'MUV' or method == 'DUD-E', "Only MUV or DUD-E molecular "
    "descriptors supported"
    n_features = 6 if method == 'DUD-E' else 17
    features = np.zeros((num_mols, n_features))
    for molname in mol_list:
        base,ext = os.path.splitext(molname)
        assert ext != '.gninatypes', "Sorry, no gninatypes support currently. "
        "Just pass the starting structure files; if you have multi-model SDFs, "
        "repeat the filename for each example derived from that file and the "
        "script will handle it."
        if ext == '.gz':
            base,ext = os.path.splitext(base)
            assert ext == '.sdf', "Only SDFs can be gzipped for now."
        if ext == '.pdb':
            mol = Chem.MolFromPDBFile(molname)
            mols = [mol]
        elif ext == 'mol2':
            mol = Chem.MolFromMol2File(molname)
            mols = [mol]
        elif ext == '.sdf':
            mols = Chem.ForwardSDMolSupplier(molname)
        elif ext == '.smi':
            with open(molname, 'r') as f:
                mols = [Chem.MolFromSmiles(line.strip()) for line in f]
        for i,mol in enumerate(mols):
            if mol is None:
                print("Problem with molecule %d from file %s" %(i+1,molname))
                failures.append(count)
            else:
                if method == 'DUD-E':
                    desc = dude_descriptors(mol)
                elif method == 'MUV':
                    desc = muv_descriptors(mol)
                else:
                    # shouldn't get here because we already asserted above,
                    # buuuuut just in case
                    print("Unsupported molecular descriptor set %s" %method)
                    sys.exit()
            features[i,:]
            count += 1

    assert num_mols == count - len(failures), "len(labels) len(mols) mismatch; "
    "ensure every mol has its own line in input types file. "

# use sklearn to cross validate, train, and optimize hyperparameters 
def fit_and_cross_validate_model(X_train, y_train, fold_it):
    '''
    Parameters
    ----------
    X_train: array_like
        training examples
    y_train: array_like
        training labels
    fold_it: iterable
        iterable yielding fold indices

    Returns
    ----------
    model: object
        fit model 
    '''

if __name__=='__main__':
    parser = ArgumentParser(description='Train a Random Forest model for '
            'classification or regression using simple molecular descriptors')
    parser.add_argument('-p', '--prefix', type=str, required=True,
            help="Prefix for training/test files: <prefix>[train|test][num].types")
    parser.add_argument('-n', '--foldnums', type=str, required=False, default=None, 
            help="Fold numbers to run, default is to determine using glob")
    parser.add_argument('-c', '--columns', type=str, default='Label,Affinity,Recfile,Ligfile'
            help='Comma-separated list of column identifiers for folds files, '
            'default is "Label,Affinity,Recfile,Ligfile"')
    parser.add_argument('-r', '--data_root', type=str, default='', 
            help='Common path to join with molnames to generate full location '
            'of files')
    parser.add_argument('-f', '--fit_all', action='store_true', 
            help='Fit all numerical columns in types file with RFRegressor. '
            'Default is to use only affinity if available, label if it is not')
    parser.add_argument('-o', '--outprefix', type=str, default='', 
            help='Output prefix for trained random forest pickle and train/test figs')
    args= parser.parse_args()

    # dump(clf, 'filename.joblib')
