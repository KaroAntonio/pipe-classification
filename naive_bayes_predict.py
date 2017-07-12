import NaiveBayes
from fuzzylogic import *
from run_naive_bayes import *

# a script to consolidate training and prediction of naive bayes

TRAINING_FID = 'data/csv/all_data.csv'
OUT_FID = 'out/naive_bayes_mitigation.csv'




# These are the methods to prep and run naive bayes

data = load_data( TRAINING_FID )

p = get_mitigation_params()

naive_bayes_accuracy(p,data)

row_fids = [row['FID'] for row in data]
cols = ['FID']

model = label_predictions(p,data)
cols += [
	p['var_map']['out'],
	p['model_name']+'_nb_pred'
	]

for row,row_fid in zip(data,row_fids):
	row['FID'] = row_fid

format_save_data(p,data,cols,out_fid=OUT_FID)
