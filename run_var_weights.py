import random

from utils import *
from params import * 

import ml_models.NaiveBayes
from naive_bayes_utils import *
from assess_naive_bayes import *

DATA_FID = 'data/csv/all_data.csv'
VAR_MAP_FID = 'var_maps/mitigation_var_map.json'

data = load_prep_data( DATA_FID )
p = get_mitigation_params()
p['var_map_fid'] = VAR_MAP_FID
p = prep_params( p )

var_weights = determine_var_weights(p, data)
print var_weights
save_data('out/var_weights.csv', [var_weights])
