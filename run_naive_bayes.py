import NaiveBayes
from fuzzylogic import *
#from params import *
import json
import sys
import copy

def count_numbers(data,attr_name):
	'''
	return: the number of values that can be converted via float(val)
	'''
	count = 0
	for row in data:
		try:
			val = row[attr_name]
			float(val)
			count+=1
		except:
			pass
	return count

def get_number_vals(data,attr_name):
	'''
	return a list of values for attr_name if all vals are numbers
	else return None
	'''
	
	if count_numbers(data,attr_name) > len(data) * 0.6:
		vals = []
		for row in data:
			try:
				val = float(row[attr_name])
			except:
				val = -1
			vals += [val]
		return vals
	else: return None


def bucketize_data_attrs(data, n=10):
	'''
	for each attribute, adjust it to fall into n buckets 
		if the values for that attr are all numbers (float/int)
	'''
	for attr_name in data[0]:
		vals = get_number_vals(data,attr_name)
		if vals:
			val_max,val_min = max(vals),min(vals)
			val_range = val_max-val_min
			bucket_size = val_range / float(n)
			if val_range > 0:
				for i,row in enumerate(data):
					row = data[i]
					row[attr_name] = int(vals[i]/bucket_size)*bucket_size

	return data

def get_instances(data, attr_names, label_name):
	'''
	return a list of {'attributes':{attr:val,...},'label','count':n}
	'''
	attr_labels = {}
	for row in data:
		attrs = {}
		row_id = 'id_'
		for name in attr_names:
			attrs[name] = row[name]
			row_id += str(row[name]) + '_'
		row_id += '_' + str(row[label_name])
		if row_id in attr_labels:
			attr_labels[row_id]['cases'] += 1
		else:
			attr_labels[row_id] = {
					'attributes':attrs,
					'label':'out={}'.format(row[label_name]),
					'cases':1}

	return list(attr_labels.values())

def build_target_col(p,data):
	for row in data:
		row['model_target'] =  p['out_f'](p,row)

def split_train_test(insts, train_ratio=0.7):

	n_train = int(train_ratio * len(insts))

	train_insts = insts[:n_train]
	test_insts = insts[n_train:]
	
	return train_insts,test_insts

def pad_values(values, target_len):
	return values + ([values[-1]]*(target_len - len(values)))

def get_padding_instances(attr_values,label_values):
	'''
	attr_values: dict of {attr1:[values,..],attr2:[values,..]}
	label_values: list of [values,...]

	return a set of instances such that every potential 
		attr and label is included
	'''
	max_len = max([len(attr) for attr in attr_values]+[len(label_values)])

	label_values = pad_values(label_values,max_len)
	for attr,values in attr_values.items():
		attr_values[attr] = pad_values(values,max_len)
	
	pad_insts = []
	for i in range(len(label_values)):
		inst = {'attributes':{a:attr_values[a][i] for a in attr_values},
				'label':'out={}'.format(label_values[i]),
				'cases':1}
		pad_insts += [inst]
	return pad_insts

def get_attr_values(data, attr_name):
	'''
	assuming data has been bucketed
	'''
	try: return [float(row[attr_name]) for row in data]
	except: return [row[attr_name] for row in data]


def attribute_weights(model, n=1000):
	''' 
	model: an nb model
	n: number of experiments
	performs some statistical analysis to determine weight relevance
	returns a dictionary of the weight of each attribute in determining class
	'''
	for i in range(n): 
		# generate a random datapoint with valid attrs
		dp = {}
		#dp['attributes'] = {k:random.choice(model.attvals[]) for k in model.attvals}
		dp['cases'] = 1



def label_predictions(p,data):

	#build_target_col(p,data)
	data = bucketize_data_attrs(data)
	attrs = p['var_map'].values()
	
	# split data into 10 train, predict batches
	window = int(len(data) / 10.)
	for i in range(10):
		predict_data = data[i*window:(i+1)*window]
		train_data = data[:i*window] + data[(i+1)*window:]

		model = build_train_model(p,data,train_data)

		for row in predict_data: 
			attr_vals = {}
			for name in attrs:
				attr_vals[name] = row[name]

			row['attributes'] = attr_vals
			row['label'] = 'out='+str(row['model_target'])
			row['cases'] = 1

			pred = model.predict(row)

			# Choose Top Label
			pred_label = ''
			max_conf = float('-inf') # confidence
			for label in pred:
				if pred[label] > max_conf:
					max_conf = pred[label]
					pred_label = label
			
			out = float(pred_label.split('=')[-1])
			row[p['model_name']+'_pred'] = out
	
	# returns the last model (of 10)
	return model

def build_train_model(p, data, train_data):

	attrs = p['var_map'].values()
	train_insts = get_instances(train_data,attrs,'model_target')

	model = NaiveBayes.NaiveBayes()	

	# PAD MODEL Instances 
	attr_values = {attr:list(set(get_attr_values(data,attr))) 
					for attr in attrs}
	label_values = list(set(get_attr_values(data,'model_target')))
	pad_insts = get_padding_instances(attr_values, label_values)
	# hacky way to assure all insts are seen at least once

	for row in pad_insts:
		model.add_instances(row)

	# TRAIN Model
	for row in train_insts:
		model.add_instances(row)

	model.train()

	return model

def naive_bayes_accuracy(p,data):

	if 'var_map_fid' in p: 
		print(p['var_map_fid'])
		p['var_map'] = json.load(open(p['var_map_fid'], 'r'))

	if 'val_map_fid' in p:
		p['val_map'] = json.load(open(p['val_map_fid'], 'r'))
	if 'params' in p:
		p['params'] = load_model_params( p['params_fid'] )
	
	build_target_col(p,data)

	data = bucketize_data_attrs(data)
	attrs = p['var_map'].values()
	instances = get_instances(data,attrs,'model_target')
	random.shuffle(instances)

	train_insts,test_insts = split_train_test(instances, train_ratio=0.9)
	
	# BUILD Model
	model = NaiveBayes.NaiveBayes()	

	# PAD MODEL Instances 
	attr_values = {attr:list(set(get_attr_values(data,attr))) 
					for attr in attrs}
	label_values = list(set(get_attr_values(data,'model_target')))
	pad_insts = get_padding_instances(attr_values, label_values)
	# hacky way to assure all insts are seen at least once
	print('MODEL: {} NUM PAD: {}'.format(p['model_name'],len(pad_insts)))

	for row in pad_insts:
		model.add_instances(row)

	# TRAIN Model
	for row in train_insts:
		model.add_instances(row)

	model.train()

	# EVAL ACCUCCURACY
	accs = []
	target_confs = []
	preds = {}
	for row in test_insts:
		pred = model.predict(row)
		for attr in pred:
			if attr not in preds: preds[attr] = []
			preds[attr] += [pred[attr]]

		target_confs += [pred[row['label']]]

	avg_confs = {attr: sum(confs)/len(confs) for attr,confs in preds.items()}
	print('Accuracy {}'.format(sum(target_confs)/len(target_confs)))
	

def format_save_data(p,data, cols=None, out_fid='out/nb_out_data.csv'):
	if not cols:
		cols = [
				p['var_map']['out'],
				p['model_name']+'_pred'
				]

	out_data = [{k:v for  k,v in row.items() if k in cols} for row in  data]
	save_data(out_fid,out_data,cols)

def predicted_label(pred):
	# Choose Top Label
	pred_label = ''
	max_conf = float('-inf') # confidence
	for label in pred:
		if pred[label] > max_conf:
			max_conf = pred[label]
			pred_label = label
	return pred_label

def apply_matrix(mat_data,row,input_map):
	# apply matrix
	for mat_row in mat_data:
		is_match = True
		# check if all cols match
		for mat_k,data_k in input_map.items():
			if row[data_k] != mat_row[mat_k]:
				is_match = False
		if is_match:
			row[mat_k] = mat_row['Mitigation']
			return row[mat_k]

def evaluate_matrix(data):
 	
	# map of pred keys to matrix keys (should maybs be moved to a file)
	input_map = {
		'Condition':'cond',
		'performance':'perf',
		'Criticality':'crit'
	}

	mat_fid = 'data/csv/mitigation_matrix.csv'
	mat_data = load_data( mat_fid )

	params = {'cond':get_condition_params(),
		'crit':get_criticality_params(),
		'perf':get_performance_params()}
	
	# load_var_maps, build target
	for name, p in params.items():
		if 'var_map_fid' in p: 
			p['var_map'] = json.load(open(p['var_map_fid'], 'r'))
		build_target_col(p,data)

	data = bucketize_data_attrs(data)
	
	# split data into 10 train, predict batches
	window = int(len(data) / 10.)
	for i in range(10):
		predict_data = data[i*window:(i+1)*window]
		train_data = data[:i*window] + data[(i+1)*window:]


		# train models
		models = {} 
		for name,p in params.items():
			models[name] = build_train_model(p,data, train_data)

		
		# for each data point (row)
		for row in predict_data: 
			# for each model, 
			top_preds = {}
			for name in models:
				m = models[name]
				attrs = params[name]['var_map'].values()
				attr_vals = { 
						name:row[name] for name in attrs  if name != 'out'
						}
			
				row['attributes'] = attr_vals 
				row['label'] = 'out='+str(row['model_target'])
				row['cases'] = 1

				pred = m.predict(row) 
				top_preds[name] = float(predicted_label(pred).split('=')[-1])
				# mold to formt that matrix is ~ a str int
				top_preds[name] = str(int(top_preds[name]))
			mat_pred = apply_matrix(mat_data,top_preds, input_map)

			row['matrix_pred'] = mat_pred
	
if __name__ == '__main__':

	data_fid = 'data/csv/all_data.csv'
	data = load_data( data_fid )

	params = [get_condition_params(),
			get_criticality_params(),
			get_performance_params(), 
			get_mitigation_params()]

	# we save the ids because they'll get mangled by the bayesian model... 
	row_fids = [row['FID'] for row in data] 
	cols = ['FID']

	_ = evaluate_matrix(data)
	cols = ['matrix_predction']

	for p in params:
		naive_bayes_accuracy(p,data)
		model = label_predictions(p,data)
		cols += [
				p['var_map']['out'],
				p['model_name']+'_nb_pred'
				]

	for row,row_fid in zip(data,row_fids):
		row['FID'] = row_fid

	format_save_data(p,data,cols)
