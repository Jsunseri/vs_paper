#!/bin/env python
import os, math, re, joblib
from argparse import ArgumentParser

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_curve, roc_auc_score

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

from train_simpledescriptor_models import generate_descriptors, FoldData, FeaturizeOutput, classifiers, regressors, methodnames

def get_pocketome_ligname(ligfile):
    m = re.search(r'(\S+)/...._(\S+)_',ligfile)
    return m.group(2)

if __name__ == '__main__':
    parser = ArgumentParser(description='Load trained random forest and '
    'generate output csv of predictions for a provided test file.')
    parser.add_argument('-p', '--pickle', nargs='+', 
            help='Pickled dump of one or more trained sklearn models, expected to have been '
            'generated with joblib')
    parser.add_argument('-t', '--testfile', nargs='+', 
            help='One or more test files with one example per line.')
    parser.add_argument('-c', '--columns', type=str, default='Label,Affinity,Recfile,Ligfile',
            help='Comma-separated list of column identifiers for folds files, '
            'default is "Label,Affinity,Recfile,Ligfile"')
    parser.add_argument('-s', '--scorename', type=str, default='',
            help='Which column to predict; if you pass nothing, it will be '
            '"Affinity" if it is available and "Label" otherwise. Eventually '
            'multiple prediction types at a time will be supported but not yet')
    parser.add_argument('-m', '--special_moltitles', action='store_true',
            help='Use the special-cased moltitle method based on parsing '
            'filenames to be compatible with CNN results on PDBbind and '
            'CrossDocked datasets')
    parser.add_argument('-r', '--data_root', nargs='*', default=[], 
            help='Common path to join with molnames to generate full location '
            'of files; can pass multiple, which will be tried in order')
    parser.add_argument('-d', '--descriptors', type=str, default='DUDE',
            help='Descriptor set to use; options are "DUDE" or "MUV"')
    parser.add_argument('-df', '--descriptor_file', type=str, default='',
            help='Optionally provide file of precomputed molecular descriptors '
            'to save time')
    parser.add_argument('-e', '--extra_descriptors', nargs='*', default=[], 
            help='Provide additional files with precomputed descriptors')
    parser.add_argument('-eo', '--extra_only', action='store_true', help='Only '
            'fit to user-provided descriptors, without computing the DUD-E or '
            'MUV desciptors')
    parser.add_argument('-o', '--outprefix', type=str, default='', 
            help='Output basename for prediction file')
    parser.add_argument('-b', '--use_babel', action='store_true', help='Use '
            'OpenBabel instead of the RDKit for parsing molecules and generating '
            'descriptors')
    parser.add_argument('-sz', '--skip_zeros', action='store_true', help='If '
            'any column of the extra descriptors is exactly 0.0, skip that example')
    parser.add_argument('-x', '--exclude', type=str, default='',
            help='Filename specifying a list of target:moltitle pairs to exclude from predictions')
    parser.add_argument('--take_first', action='store_true',
            help='Take first mol from multi-model files.')
    args = parser.parse_args()

    if args.extra_only:
        assert args.extra_descriptors, ('Fitting exclusively to user-provided descriptors '
        'was requested, but no user-provided descriptors were provided')
        if args.descriptor_file:
            print('Descriptor file was provided, but extra_only was passed so '
                    'these descriptors will not be used.')

    for pfile in args.pickle:
        assert os.path.isfile(pfile), "%s does not exist" %pfile

    print('Parsing test molecules\n')
    column_names = [name for name in args.columns.split(',') if name]
    scorecol = args.scorename
    if not scorecol:
        if 'Affinity' in column_names:
            scorecol = 'Affinity'
        elif "Label" in column_names:
            scorecol = 'Label'
        else:
            assert 0, 'Unknown scoring target column'
    assert scorecol in column_names, 'Column provided as scoring target not in column names'
    df_list = [pd.read_csv(fname, names=column_names, delim_whitespace=True) for fname in args.testfile]
    df = pd.concat(df_list, ignore_index=True, sort=False)
    mol_list = df['Ligfile'].tolist()
    labels = df[scorecol].tolist()

    print('Checking for user-provided descriptors\n')
    # include features from user-provided file of precomputed features, if available
    extra_descs = None
    desclist = []
    skip_indices = []
    for descfile in args.extra_descriptors:
        desclist.append(pd.read_csv(descfile, delim_whitespace=True, header=None))
    if desclist:
        extra_descs = np.hstack([extra_df.to_numpy() for extra_df in desclist])
        assert extra_descs.shape[0] == len(labels), 'Extra descriptor file should have the same number of examples as the input but has %s instead of %s' %(extra_descs.shape[0], len(labels))
        if args.skip_zeros:
            skip_indices = np.nonzero(extra_descs==0.0)[0].tolist()
    if args.extra_only:
        featurize_output = FeaturizeOutput(extra_descs, [], [])
        features = featurize_output.features
        failures = featurize_output.failures
        moltitles = featurize_output.moltitles
    else:
        if args.descriptor_file:
            _,featurize_output = joblib.load(args.descriptor_file)
            if args.extra_descriptors:
                shape = featurize_output.features.shape
                extra_shape = extra_descs.shape
                assert shape[0] == extra_shape[0], ('First dim of descriptors and '
                'extra_descriptors must match, but they are %d and %d' %(shape[0], extra_shape[0]))
                allfeatures = np.append(featurize_output.features,
                        extra_descs, axis=1)
                featurize_output = FeaturizeOutput(allfeatures,
                        featurize_output.failures, featurize_output.moltitles)
            features = featurize_output.features
            failures = featurize_output.failures
            moltitles = featurize_output.moltitles
        else:
            print('Generating descriptors\n')
            # TODO: can infer which descriptors to use from model, since it was fit
            # with one of the descriptor sets, instead of making the user pass it
            features, failures, moltitles = generate_descriptors(mol_list,
                    args.data_root, args.descriptors, args.use_babel, extra_descs, args.take_first)
    if skip_indices:
        features = np.delete(features, skip_indices, 0)
        if moltitles:
            moltitles = [title for i,title in enumerate(moltitles) if i not in skip_indices]
        labels = np.delete(labels, skip_indices, axis=0)
        if failures:
            failures = [failure for failure in failures if failure not in skip_indices]
        df = df.drop(skip_indices)

    print('Making predictions\n')
    for pfile in args.pickle:
        model = joblib.load(pfile)
        # LABEL PREDICTION TARGET TITLE METHOD
        # for predictions, don't delete failures, just predict inactive (i.e. ~3?
        # for regressor, 0 for classifier)
        if issubclass(model.__class__, regressors):
            y_pred = model.predict(features)
            for failure in failures:
                y_pred = np.insert(y_pred, failure, 3.0)
        elif issubclass(model.__class__, classifiers):
            y_pred = model.predict_proba(features)[:,1]
            for failure in failures:
                y_pred = np.insert(y_pred, failure, 0.0)
        else:
            assert 0, 'Unrecognized sklearn class %s' % type(model).__name__

        method = os.path.splitext(os.path.basename(pfile))[0]
        outname = '%s%s' %(args.outprefix, method)

        df['Prediction'] = y_pred
        df['Method'] = method
        df['Target'] = df['Ligfile'].apply(lambda x: os.path.basename(os.path.dirname(x)))
        if args.special_moltitles:
            if 'PocketomeGenCross_Output' in args.data_root[0].split('/'):
                df['Title'] = df['Ligfile'].apply(lambda x: get_pocketome_ligname(x))
            else:
                df['Title'] = df['Ligfile'].apply(lambda x: os.path.basename(x).split('_')[0])
            columnames = [scorecol, 'Prediction', 'Target', 'Title', 'Method']
        elif moltitles:
            df['Title'] = moltitles
            columnames = [scorecol, 'Prediction', 'Target', 'Title', 'Method']
        else:
            print('Molecule titles not found in provided input')
            columnames = [scorecol, 'Prediction', 'Target', 'Method']
        if args.exclude:
            assert 'Title' in columnames, 'Cannot exclude based on molecule titles because molecule titles were not found'
            exclude_info = pd.read_csv(args.exclude, delim_whitespace=True,
                    header=None, names=['Target', 'Title'])
            exclude_data = set(['-'.join(item) for item in 
                zip(exclude_info['Target'].tolist(), exclude_info['Title'].tolist())])
            df['Excludekey'] = df.apply(lambda row: row['Target'] + '-' + row['Title'], axis=1)
            df = df[~(df['Excludekey'].isin(exclude_data))]
        df.to_csv('%s.csv' %outname, sep=' ', columns=columnames, index=False, header=False)
        y_true = df[scorecol].tolist()

        if scorecol == 'Label':
            # do AUC...and eventually EF1% (TODO)
            fpr,tpr,_ = roc_curve(y_true,y_pred)
            auc = roc_auc_score(y_true,y_pred)
            print('{} AUC: {:0.3f}'.format(type(model).__name__, auc))
        else:
            # do RMSE and R
            r,_ = pearsonr(labels, y_pred)
            print('{} R: {:0.3f}'.format(type(model).__name__, r))
            rmse = math.sqrt(mean_squared_error(labels, y_pred))
            print('{} RMSE: {:0.3f}'.format(type(model).__name__, rmse))
