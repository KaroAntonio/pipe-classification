import random

from core.utils import *
from core.params import * 

import models.NaiveBayes
from core.naive_bayes_utils import *

# a script to consolidate training and prediction of naive bayes

TRAINING_FID = 'data/csv/all_data.csv'
VAR_MAP_FID = 'var_maps/best_nb.json'
OUT_FID = 'out/naive_bayes_dec.csv'

model_name = 'best_nb'
train_data = load_data( TRAINING_FID )

p = prep_data_run_naive_bayes( train_data, OUT_FID, model_name )

