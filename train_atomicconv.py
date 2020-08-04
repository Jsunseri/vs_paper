import os,glob,re,joblib
from argparse import ArgumentParser

import numpy as np
import pandas as pd

import deepchem as dc
from deepchem.feat.graph_features import AtomicConvFeaturizer

def do_atomconv_featurize(lig_files, rec_files, labels):
    '''
    Parameters
    ----------
    lig_files: array_like
        n_examples list of ligand file names for training
    rec_files: array_like
        n_examples list of receptor file names for training
    labels: array_like
        n_examples list of labels

    Returns
    ----------
    features: array_like
        n_examples X feature_dims
    failures: array_like
        list of example indices that failed to featurize
    '''
    frag1_num_atoms = 150  # for ligand atoms
    frag2_num_atoms = 27000  # for protein atoms
    complex_num_atoms = frag1_num_atoms + frag2_num_atoms
    neighbor_cutoff = 4
    max_num_neighbors = 4

    featurizer = AtomicConvFeaturizer(
        labels=labels,
        frag1_num_atoms=frag1_num_atoms,
        frag2_num_atoms=frag2_num_atoms,
        complex_num_atoms=complex_num_atoms,
        neighbor_cutoff=neighbor_cutoff,
        max_num_neighbors=max_num_neighbors,
        batch_size=64)
    
    print("Featurizing Complexes")
    return featurizer.featurize_complexes(lig_files, rec_files)

def find_and_parse_folds(prefix, foldnums='',
        columns='Label,Affinity,Recfile,Ligfile', data_root=[], fit_all=False):
    '''
    Parameters
    ----------
    prefix: str
        Prefix for fold files, we'll glob for the full list
    foldnums: str, optional
        which foldnums to include; default is to glob and include all matches
    columns: str, optional
        comma-separated list of column identifiers for fold files; defaults to
        "Label,Affinity,Recfile,Ligfile"
    data_root: array_like, optional
        top-level root dir for all files, up to two can be passed
    fit_all: bool, optional
        whether to fit all numerical columns with RandomForestRegressor.
        default is to prefer affinity if available, fit classifier with label
        if it is not, and complain otherwise unless this was passed

    Returns
    ----------
    datasets: array_like
        list of dataset objects for cross validation
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
    assert files, 'No fold files matching prefix were found'

    # we *require* a recfile, ligfile, and at least one target for
    # regression/classification. for now we default to using the Affinity
    # column if present
    column_names = [name for name in columns.split(',') if name]
    assert column_names, "Missing column names for types files."
    assert 'Label' or 'Affinity' in column_names, "Currently a 'Label' or 'Affinity' column are required."
    assert 'Recfile' in column_names, "Path to receptor file required."
    assert 'Ligfile' in column_names, "Path to ligand file required."

    # parse files with pandas; since we have allowed the column layout to vary,
    # we default to the "normal" version but allow the user to specify others.
    df_list = [pd.read_csv(fname, names=column_names, delim_whitespace=True) for fname in files]

    # fill in list of tuples arranged like [(train0, test0), (train1, test1), ..., (trainN, testN)]
    # where the elements are arrays of indices associated with these folds
    elems_per_df = [df.shape[0] for df in df_list]
    total_examples = sum(elems_per_df)
    print('Got %d examples from %d files\n' %(total_examples, len(files)))

    fold_datasets = []
    # set up NumpyDatasets for each fold
    for df in df_list:
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

        # sigh, this is because some of David's pdbbind directories are missing
        # content
        df['Recfile'] = df['Recfile'].apply(lambda x: 
                os.path.join(data_root[0],x) if
                os.path.isfile(os.path.join(data_root[0],x)) else 
                os.path.join(data_root[1],x))
        df['Ligfile'] = df['Ligfile'].apply(lambda x:
                os.path.join(data_root[0],x) if
                os.path.isfile(os.path.join(data_root[0],x)) else
                os.path.join(data_root[1], x))
        rec_files = df['Recfile'].to_numpy()
        lig_files = df['Ligfile'].to_numpy()
        # generate features
        features, failures = do_atomconv_featurize(lig_files, rec_files, labels)
        print('Featurization complete with %d failures.'%len(failures))
        labels = np.delete(labels, failures)

        # construct dataset 
        fold_datasets.append(deepchem.data.DiskDataset.from_numpy(features, y=labels))
    return fold_datasets

if __name__=='__main__':
    parser = ArgumentParser(description='Train DeepChem AtomConv model using pre-generated data folds')
    parser.add_argument('-p', '--prefix', type=str, required=True,
            help="Prefix for training/test files: <prefix>[train|test][num].types")
    parser.add_argument('-n', '--foldnums', type=str, required=False, default=None, 
            help="Fold numbers to run, default is to determine using glob")
    parser.add_argument('-c', '--columns', type=str, default='Label,Affinity,Recfile,Ligfile',
            help='Comma-separated list of column identifiers for folds files, '
            'default is "Label,Affinity,Recfile,Ligfile"')
    parser.add_argument('-r', '--data_root', nargs='*', default=[], 
            help='Common path to join with molnames to generate full location '
            'of files')
    parser.add_argument('-s', '--seed', type=int, default=42,
            help='Random seed, used for boostrapping and feature sampling')
    parser.add_argument('-f', '--fit_all', action='store_true', 
            help='Fit all numerical columns in types file with RFRegressor. '
            'Default is to use only affinity if available, label if it is not')
    parser.add_argument('-a', '--atomconv_features', default='',
            help='Precomputed AtomConv features for this dataset')
    parser.add_argument('-o', '--outprefix', type=str, default='AtomConv_', 
            help='prefix for all output files')
    args= parser.parse_args()
   
    np.random.seed(0)
    if not args.data_root:
        args.data_root = ['']
    # foldfiles = {
            # 'refined':
            # '/net/pulsar/home/koes/jss97/git/gnina_models/data/PDBBind2016/Refined_types/ccv_ref_uff_0_', 
            # 'general':
            # '/net/pulsar/home/koes/jss97/git/gnina_models/data/PDBBind2016/General_types/ccv_gen_uff_0_', 
            # 'crossdock':
            # '/net/pulsar/home/koes/paf46/Research/affinity_paper/it2_tt_0_'
            # }

    failed = False
    if args.atomconv_features:
        try:
            fold_data = joblib.load(args.atomconv_features)
        except Exception as e:
            print(e)
            failed = True
    if not args.atomconv_features or failed:
        # get data and fold partitions
        fold_data = find_and_parse_folds(args.prefix, args.foldnums, args.columns, args.data_root, args.fit_all)
        print('Dumping AtomConv features for future reuse.')
        joblib.dump(fold_data, '%s%dfold_features.joblib' %(args.outprefix, len(fold_data)))

    # TODO hyperparam opt? deepchem provides analogous utilities to sklearn
    # for this
    # TODO? the dc examples balance classes by adding sample weights so that
    # the sum of all example weights from each class is the same
    columns = [name for name in args.columns.split(',') if name]
    metrics = []
    # TODO: fix
    assert not args.fit_all and 'Affinity' in columns or 'Label' in columns, 'Currently just '
    'supporting individual regression or classification tasks'
    if args.fit_all:
        for column in columns:
            # i don't think mixing task modes like this actually works
            if column == 'Label':
                metrics.append(dc.metrics.Metric(dc.metrics.roc_auc_score,
                    mode="classification"))
            else:
                metric.append(dc.metrics.Metric(dc.metrics.pearson_r2_score,
                    mode='regression'))
    elif 'Affinity' in columns:
        metric.append(dc.metrics.Metric(dc.metrics.pearson_r2_score, mode='regression'))
    else:
        assert 'Label' in columns, 'Unless fit_all is passed, "Label" or '
        '"Affinity" columns are currently required'
        metrics.append(dc.metrics.Metric(dc.metrics.roc_auc_score,
            mode="classification"))
    
    frag1_num_atoms = 150  # for ligand atoms
    frag2_num_atoms = 27000  # for protein atoms
    complex_num_atoms = frag1_num_atoms + frag2_num_atoms
    model = dc.models.AtomicConvModel(
        frag1_num_atoms=frag1_num_atoms,
        frag2_num_atoms=frag2_num_atoms,
        complex_num_atoms=complex_num_atoms,
        model_dir=os.getcwd())
    
    # Fit trained model
    for i in range(len(fold_data)):
        test_dataset = fold_datasets[i]
        train_folds = [fold_data[j] for j in range(len(fold_data)) if j != i]
        train_dataset = dc.splits.merge_fold_datasets(train_folds)

        print("Fitting model on train dataset")
        model.fit(train_dataset)
        model.save() # pretty sure this is unnecessary
        
        print("Evaluating model")
        train_scores = model.evaluate(train_dataset, metrics)
        test_scores = model.evaluate(test_dataset, metrics)
        try:
            cv_scores = np.concatenate(cv_scores, test_scores)
        except NameError:
            cv_scores = test_scores
        
print("Mean Scores on evaluation dataset")
print(np.mean(cv_scores))
print("Standard Deviations on evaluation dataset")
print(np.std(std_scores))
