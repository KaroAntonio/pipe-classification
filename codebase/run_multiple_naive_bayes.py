import random

from core.utils import *
from core.params import * 

import models.NaiveBayes
from core.naive_bayes_utils import *

# a script to consolidate training and prediction of naive bayes

TRAINING_FID = 'data/csv/all_data.csv'
OUT_FID = 'out/naive_bayes_{}_2019.csv'
MODEL_NAMES = [
	#'performance',
	#'criticality',
	#'condition',
	'mitigation'
	]

def output_joint_models( res ):
  # build fid_map
  # join the naive bayes predictions for all models into one dataset
  fid_map = {}
  for model_name in MODEL_NAMES:
    params = {'model_name':model_name}
    var_map_fid = 'data/var_maps/{}_var_map.json'.format(model_name)
    params['var_map_fid'] = var_map_fid
    params = prep_params( params )

    out_col = params['out'] 
    pred_col = '{}_pred'.format(model_name)

    for d in res[model_name]['data']:
      fid = d['FID']
      if fid not in fid_map: 
        fid_map[fid] = {}
        fid_map[fid]['FID'] = d['FID']

      fid_map[fid][out_col] = d[out_col]
      fid_map[fid][pred_col] = d[pred_col]

  return fid_map

res = {}
train_data = load_data( TRAINING_FID )
for model_name in MODEL_NAMES:
  var_map_fid = 'data/var_maps/{}_var_map.json'.format(model_name)
  res_out_fid = OUT_FID.format(model_name)
  res[model_name] = prep_data_run_naive_bayes( train_data, res_out_fid, model_name, save=False )

print('Saving All Model Output: {}'.format('out/naive_bayes_all_models.csv'))
fid_map = output_joint_models( res )
fieldnames = ['FID','TARGET_Performance_Level','performance_pred','TARGET_Criticality_Level','criticality_pred','condition_pred','TARGET_Condition_level','TARGET_MITIGATION_Level_PAN','mitigation_pred']
save_data('out/naive_bayes_all_models.csv',fid_map.values(), fieldnames)

