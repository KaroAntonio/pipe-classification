import NaiveBayes
from fuzzylogic import *
import json
import sys

def count_numbers(data,attr_name):
	'''
	return: the number of values that can be converted via float(val)
	'''
	count = 0
	for row in data:
		val = row[attr_name]
		try:
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

def label_predictions(p,data):

	#build_target_col(p,data)
	data = bucketize_data_attrs(data)
	attrs = p['var_map'].values()
	
	# split data into 10 train, predict batches
	window = int(len(data) / 10.)
	train_pred_batches = []
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
	
			print(pred_label)
			out = float(pred_label.split('=')[-1])
			row[p['model_name']+'_pred'] = out

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
		p['var_map'] = json.load(open(p['var_map_fid'], 'r'))

	if 'val_map_fid' in p:
		p['val_map'] = json.load(open(p['val_map_fid'], 'r'))
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
	print('NUM PAD:',len(pad_insts))

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
	print('Accuracy',sum(target_confs)/len(target_confs))

def format_save_data(p,data):
	cols = [
            p['var_map']['out'],
            p['model_name']+'_pred'
            ]

	out_data = [{k:v for  k,v in row.items() if k in cols} for row in  data]
	save_data('out/nb_out_data.csv',out_data,cols)

if __name__ == '__main__':
	if sys.version_info[0] == 2:
		print('YO USE PYTHON 3')

	data_fid = 'data/csv/all_data.csv'
	data = load_data( data_fid )

	p = get_condition_params()
	#p = get_criticality_params()
	#p = get_performance_params()

	naive_bayes_accuracy(p,data)
	label_predictions(p,data)
	
	format_save_data(p,data)
