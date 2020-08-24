#!/bin/env python
import os, math, joblib
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

from train_simpledescriptor_models import generate_descriptors, FeaturizeOutput, classifiers, regressors, methodnames

if __name__ == '__main__':
    parser = ArgumentParser(description='Load trained random forest and '
    'generate output csv of predictions for a provided test file.')
    parser.add_argument('-p', '--pickle', type=str, required=True, 
            help='Pickled dump of trained random forest, expected to have been '
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
    parser.add_argument('-r', '--data_root', nargs='*', default=[], 
            help='Common path to join with molnames to generate full location '
            'of files; can pass multiple, which will be tried in order')
    parser.add_argument('-m', '--method', type=str, default='DUD-E',
            help='Descriptor set to use; options are "DUD-E" or "MUV"')
    parser.add_argument('-o', '--outname', type=str, default='', 
            help='Output basename for prediction file')
    args = parser.parse_args()

    assert os.path.isfile(args.pickle), "%s does not exist" %args.pickle
    rf = joblib.load(args.pickle)

    print('Parsing test molecules\n')
    mol_list = []
    labels = []
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
    for fname in args.testfile:
        df = pd.read_csv(fname, names=column_names, delim_whitespace=True)
        this_mol_list = df['Ligfile'].tolist()
        these_labels = df[scorecol].tolist()
        mol_list += this_mol_list
        labels += these_labels

    print('Generating descriptors\n')
    features, failures, moltitles = generate_descriptors(mol_list, args.data_root, args.method)
    if not moltitles:
        print('Molecule titles not found in provided input')

    print('Making predictions\n')
    # LABEL PREDICTION TARGET TITLE METHOD
    # for predictions, don't delete failures, just predict inactive (i.e. ~3?
    # for regressor, 0 for classifier)
    if isinstance(rf, RandomForestRegressor):
        y_pred = rf.predict(features)
        for failure in failures:
            y_pred = np.insert(y_pred, failure, 3.0)
    elif isinstance(rf, RandomForestClassifier):
        y_pred = rf.predict_proba(features)[:,1]
        for failure in failures:
            y_pred = np.insert(y_pred, failure, 0.0)
    else:
        assert 0, 'Unrecognized Random Forest class %s' %type(rf)

    outname = args.outname
    if not outname:
        outname = 'rf_preds'

    df['Prediction'] = y_pred
    df['Method'] = 'RF_%s' %args.method
    df['Target'] = df['Ligfile'].apply(lambda x: os.path.basename(os.path.dirname(x)))
    if moltitles:
        df['Title'] = moltitles
        columnames = ['Label', 'Prediction', 'Target', 'Title', 'Method']
    else:
        columnames = ['Label', 'Prediction', 'Target', 'Method']
    df.to_csv('%s.csv' %outname, sep=' ', columns=columnames, index=False, header=False)
    y_true = df['Label'].tolist()

    if scorecol == 'Label':
        # do AUC...and eventually EF1% (TODO)
        fpr,tpr,_ = roc_curve(y_true,y_pred)
        auc = roc_auc_score(y_true,y_pred)
        print('AUC: {:0.3f}'.format(auc))
    else:
        # do RMSE and R
        r,_ = pearsonr(labels, y_pred)
        print('R: {:0.3f}'.format(r))
        rmse = math.sqrt(mean_squared_error(labels, y_pred))
        print('RMSE: {:0.3f}'.format(rmse))
