import random

from utils import *
from params import * 

import ml_models.NaiveBayes
from naive_bayes_utils import *

# a script to consolidate training and prediction of naive bayes

TRAINING_FID = 'data/csv/final_data_december_sample.csv'
VAR_MAP_FID = 'var_maps/dec_var_map.json'
OUT_FID = 'out/naive_bayes_dec.csv'

p = prep_data_run_naive_bayes( TRAINING_FID, OUT_FID, VAR_MAP_FID )

