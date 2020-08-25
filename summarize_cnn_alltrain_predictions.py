#!/bin/env python
import sys,os,re
import pandas as pd

# we have three distinct (architecture + training set) models
basedir='/net/pulsar/home/koes/jss97'
prefixes = ['/datasets/crossdocked/cd_', '/datasets/dense/dense_cd_',
            '/datasets/pdbbind-general-2016/gen_']
name_map = {'/datasets/crossdocked/cd_': 'crossdocked_default2018-',
            '/datasets/dense/dense_cd_': 'dense-',
            '/datasets/pdbbind-general-2016/gen_': 'general_default2018-'}
labeltypes = ['score', 'affinity']

for prefix in prefixes:
  data = {}
  colnames = {}
  # each was trained with 5 different seeds, these should be the rows
  data['lig'] = []
  data['rec'] = []
  data['True_label'] = []
  data['True_affinity'] = []
  for seed in range(5):
    for scoretype in labeltypes:
      colname = scoretype + '_' + str(seed)
      data[colname] = []
      try:
        colnames[scoretype].append(colname)
      except KeyError:
        colnames[scoretype] = [colname]
    # we made predictions for 3 folds that should be concatenated
    for fold in range(3):
      fname = '%s%ss%df%d.predictions' %(basedir, prefix, seed, fold)
      with open(fname, 'r') as f:
        for line in f:
          if line.startswith('#'):
              continue
          contents = line.split()
          # if it's the dense net it's missing the RMSD (and the comment at the
          # line end, making it distinguishable by number of elements)
          pred_label = float(contents[0])
          pred_aff = float(contents[1])
          label = int(contents[3])
          aff = float(contents[4])
          if aff < 0:
            aff = -aff
          if len(contents) == 10 or len(contents) == 9:
            rmsd = float(contents[5])
            rec = contents[6]
            lig = contents[7]
          else:
            rec = contents[5]
            lig = contents[6]
          data['score_' + str(seed)].append(pred_label)
          data['affinity_' + str(seed)].append(pred_aff)
          if seed == 0:
            if prefix == '/datasets/pdbbind-general-2016/gen_':
              rec_name = rec.split('/')[-1].split('_')[0]
              lig_name = os.path.basename(lig).split('_')[0]
            else:
              m = re.search(r'(\S+)/...._._rec_...._(\S+)_(lig|uff_min)',lig)
              rec_name = m.group(1) # really the pocket
              lig_name = m.group(2)
            data['lig'].append(lig_name)
            data['rec'].append(rec_name)
            data['True_label'].append(label)
            data['True_affinity'].append(aff)
  # then we want to take the mean over the ensemble for each model
  df = pd.DataFrame(data)
  for scoretype in labeltypes:
    mean = scoretype + '-mean'
    df[mean] = df[colnames[scoretype]].mean(axis=1)
    # then make predictions per protein+ligand pair 
    grouped = df.groupby(['lig', 'rec'], as_index=False)
    pred = df.loc[grouped[mean].idxmax()]
    method = '%sCNN%s-mean' %(name_map[prefix],scoretype)
    pred['Method'] = method
    pred.to_csv(path_or_buf='%s.csv' %(method), sep=' ', header=False, index=False,
            columns=['True_affinity', mean, 'rec', 'lig', 'Method'])
    # if affinity, also use the label to rank and then output the associated
    # affinity
    if scoretype == 'affinity':
        # column should exist because we do the label first
        pred = df.loc[grouped['score-mean'].idxmax()]
        pred['Method'] = method 
        pred.to_csv(path_or_buf='%s_scorerank.csv' %(method), sep=' ', header=False, index=False,
                    columns=['True_affinity', mean, 'rec', 'lig', 'Method'])
