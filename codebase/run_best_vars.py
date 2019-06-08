import random

from core.utils import *
from core.params import * 
from core.naive_bayes_utils import *
from core.assess_naive_bayes import *

DATA_FID = 'data/csv/all_data.csv'
VAR_MAP_FID = 'data/var_maps/mitigation_var_map.json'
OUT_FID = 'best_nb.json'

data = load_prep_data( DATA_FID )
p = get_mitigation_params()
p['var_map_fid'] = VAR_MAP_FID
p = prep_params( p )

determine_save_best_vars(p,data,OUT_FID)

