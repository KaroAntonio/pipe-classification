import random

from utils import *
from params import * 

import ml_models.NaiveBayes
from naive_bayes_utils import *

# a script to consolidate training and prediction of naive bayes

TRAINING_FID = 'data/csv/all_data.csv'
OUT_FID = 'out/naive_bayes_mitigation.csv'

# These are the methods to prep and run naive bayes

data = load_data( TRAINING_FID )
random.shuffle(data)


# save FIDs after shuffling so that the order is preserved
row_fids = [row['FID'] for row in data]

c_data = clean_data( data )
b_data = bucketize_data( c_data ) 
data = b_data

p = prep_params( get_mitigation_params() )

for row in data:
	row[p['out']] = random.choice(range(1,4))

acc = label_predictions(p,data)

print('{} acc: {} '.format(p['model_name'],acc))

cols = ['FID']
cols += [
	p['var_map']['out'],
	p['model_name']+'_pred'
	]

# restore FIDs
for row,row_fid in zip(data,row_fids):
	row['FID'] = row_fid

print('Output Distribution:')
print(get_attr_val_counts(data, p['out']))
print('Prediction Distribution:')
print(get_attr_val_counts(data, 'mitigation_model_pred'))

format_save_data(p,data,cols,out_fid=OUT_FID)
'''

# a quick validation
with open('out/naive_bayes_mitigation.csv') as f:
	n_match = 0
	lines = f.readlines()
	for line in lines[1:]:
		parts = line.split(',')
		if parts[1].strip() == parts[2].strip():
			n_match += 1
	acc = float(n_match) / (len(lines)-1)
	print(acc)
'''

		
