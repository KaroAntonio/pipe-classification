import sys
from fuzzylogic import * 
from MatLinearRegression import LinearRegression
import random

def vectorize_data(p):
	'''	
	convert that data to number only values
	'''
	pass
	
def build_xy(p,n=None):
	'''
	p: params
	n: num samples
	return x and y for training in np format
	'''
	df = p['df']
	for k in df:
		random.shuffle(df[k])

	x = []
	y = []
	if not n: n = float('inf')
	for i in range(min(n,len(df[df.keys()[0]]))):
		feats = []
		for k in df:
			feats += [df[k][i]]
		x += [feats]
		y += [df[p['var_map']['out']][i]]

	return x,y 

def format_save(params, fid):

	data = params[0]['data']
	cols = ['FID']
	for p in params:
		cols += [
				p['var_map']['out'],
				p['model_name']+'_pred'
				]

	out_data = [{k:v for  k,v in row.items() if k in cols} for row in  data]
	save_data(fid,out_data,cols)

def get_pred(p,data):

	p['data'] = data

	if 'var_map_fid' in p: p['var_map'] = json.load(open(p['var_map_fid'], 'r'))
	if 'val_map_fid' in p: p['val_map'] = json.load(open(p['val_map_fid'], 'r'))

	df = build_data_frame(p)
	p['df'] = df

	# Display Data 
	for k in df:
		print('{} max: {}, min: {}'.format(k,max(df[k]),min(df[k])))
	
	X,y = build_xy(p, 10)
	print('X y built: {} samples'.format(len(X)))
	train_ratio = 0.9 # ratio of samples used for training
	n_train = int(len(X) * train_ratio)
	X_train, X_test = X[:n_train], X[n_train:]
	y_train, y_test = y[:n_train], y[n_train:]

	y_train, y_test = [y_train], [y_test]

	model = LinearRegression() 

	print('Training...')
	trained_model = model.fit(X_train, y_train)

	print('Evaluating...')
	print(trained_model.score(X_test,y_test))

	return trained_model.predict(X)

def add_data_regression_preds(params,data):
	'''
	adds the pred for each row to the data
	params: a  list of params
	'''
	preds = []
	for p in params:
		pred = get_pred(p,data)
		for i,e in enumerate(pred):
			data[i][p['model_name']+'_pred'] = float(e)

if __name__ == '__main__':
	data_fid = 'data/csv/all_data.csv'
	data = load_data( data_fid )
	random.shuffle(data)
	
	params = [
		#get_condition_params(),
		#get_criticality_params(),
		#get_performance_params()
		get_mitigation_params()
		]

	add_data_regression_preds(params,data)
	format_save(params,'out/reg_out_data.csv')

