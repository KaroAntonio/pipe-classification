import random

from utils import *
from params import * 

import ml_models.NaiveBayes
from naive_bayes_utils import *

# a script to consolidate training and prediction of naive bayes

TRAINING_FID = 'data/csv/all_data.csv'
OUT_FID = 'out/naive_bayes_{}_2019.csv'
MODEL_NAMES = [
	'performance',
	'criticality',
	'condition',
	'mitigation'
	]

def output_joint_models( res ):
  # build fid_map
  fid_map = {}
  for model_name in MODEL_NAMES:
    out_col = '{}_Level_PAN'.format(model_name.upper())  
    pred_col = '{}_model_pred'.format(model_name)
    for d in res[model_name]['data']:
      fid = d['FID']
      if fid not in fid_map: 
        fid_map[fid] = {}
        fid_map[fid]['FID'] = d['FID']
      # the out cols are referred to as mitigation in all cases, for whatever reason
      fid_map[fid][out_col] = d['MITIGATION_Level_PAN']
      fid_map[fid][pred_col] = d['mitigation_model_pred']

  return fid_map

res = {}
for model_name in MODEL_NAMES:
    var_map_fid = 'var_maps/{}_var_map.json'.format(model_name)
    res_out_fid = OUT_FID.format(model_name)
    res[model_name] = prep_data_run_naive_bayes( TRAINING_FID, res_out_fid, var_map_fid )


print('Saving All Model Output: {}'.format('out/naive_bayes_all_models.csv'))
fid_map = output_joint_models( res )
fieldnames = ['FID','PERFORMANCE_Level_PAN','performance_model_pred','CRITICALITY_Level_PAN','criticality_model_pred','condition_model_pred','CONDITION_Level_PAN','MITIGATION_Level_PAN','mitigation_model_pred']
save_data('out/naive_bayes_all_models.csv',fid_map.values(), fieldnames)

