import random

from utils import *
from params import * 

import ml_models.NaiveBayes
from naive_bayes_utils import *

# a script to consolidate training and prediction of naive bayes

TRAINING_FID = 'data/csv/all_data.csv'
OUT_FID = 'out/naive_bayes_dec.csv'
MODEL_NAMES = [
	'performance',
	'criticality',
	'condition',
	'mitigation',
	]

for model_name in MODEL_NAMES:
	var_map_fid = 'var_maps/{}_var_map.json'.format(model_name)
	p = prep_data_run_naive_bayes( TRAINING_FID, OUT_FID, var_map_fid )

