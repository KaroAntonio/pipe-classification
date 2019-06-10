import random

from core.utils import *
from core.params import * 

import models.NaiveBayes
from core.naive_bayes_utils import *

# Script to run assessment on vars to determine which contributes most to prediction

DATA_FID = 'data/csv/all_data.csv'
VAR_MAP_FID = 'data/var_maps/mitigation_var_map.json'
OUT_FID = 'out/var_weights.csv'

data = load_prep_data( DATA_FID )
p = get_mitigation_params()
p['var_map_fid'] = VAR_MAP_FID
p = prep_params( p )

var_weights = determine_var_weights(p, data)
print var_weights
save_data(OUT_FID, [var_weights])
