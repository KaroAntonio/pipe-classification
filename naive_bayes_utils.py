import random
import json
import sys
import copy

import ml_models.NaiveBayes as NaiveBayes
from params import *
from utils import *

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
	for i in range(max_len):
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
	attrs = p['var_map'].values()
	
	# split data into 10 train, predict batches
	window = int(len(data) / 10.)
	n_correct_preds = 0
	for i in range(10):

		predict_data = data[i*window:(i+1)*window]
		if (i == 9): predict_data += data[(i+1)*window:]
		train_data = data[:i*window] + data[(i+1)*window:]

		model = build_train_model(p,data,train_data)

		for row in predict_data: 
			attr_vals = {}
			for name in attrs:
				if name != p['out']:
					attr_vals[name] = row[name]

			row['attributes'] = attr_vals
			row['label'] = 'out='+str(row[p['out']])
			row['cases'] = 1
	
			try:
				pred = model.predict(row)
			except:
				print('FAILED',attr_vals, row['label'])
	
			pred_label = predicted_label(pred)
			
			out = float(pred_label.split('=')[-1])
			row[p['model_name']+'_pred'] = out
			n_correct_preds += 1 if out == row[p['out']] else 0
	
	# returns accuracy 
	return n_correct_preds / float(len(data) )

def build_train_model(p, data, train_data):

	attrs = [v for k,v in p['var_map'].items() if k != 'out']
	train_insts = get_instances(train_data,attrs,p['out'])

	model = NaiveBayes.NaiveBayes()	

	# PAD MODEL Instances 
	attr_values = {
		attr:list(set(get_attr_values(data,attr))) 
		for attr in attrs
	}

	label_values = list(set(get_attr_values(data,p['out'])))
	pad_insts = get_padding_instances(attr_values, label_values)
	# hacky way to assure all insts are seen at least once

	# TRAIN padding instances 
	for row in pad_insts:
		model.add_instances(row)

	# TRAIN Model
	for row in train_insts:
		model.add_instances(row)

	model.train()

	return model

def naive_bayes_accuracy(p,data):

	attrs = p['var_map'].values()

	instances = get_instances(data,attrs,p['out'])
	print(len(instances))

	train_insts,test_insts = split_train_test(instances, train_ratio=0.9)
	
	# BUILD Model
	model = NaiveBayes.NaiveBayes()	

	# PAD MODEL Instances 
	attr_values = {attr:list(set(get_attr_values(data,attr))) 
					for attr in attrs}
	label_values = list(set(get_attr_values(data,p['out'])))

	pad_insts = get_padding_instances(attr_values, label_values)

	# hacky way to assure all insts are seen at least once
	for row in pad_insts:
		model.add_instances(row)

	# TRAIN Model
	for row in train_insts:
		model.add_instances(row)

	model.train()

	print(len(test_insts))

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
	
	print('n rows',len(target_confs))
	return sum(target_confs)/len(target_confs)
	
def predicted_label(pred):
	# Choose Top Label
	pred_label = ''
	max_conf = float('-inf') # confidence
	for label in pred:
		if pred[label] > max_conf:
			max_conf = pred[label]
			pred_label = label
	return pred_label

def prep_data_run_naive_bayes( train_fid, out_fid, var_map_fid, save=True ):

	# These are the methods to prep and run naive bayes
	data = load_data( train_fid )
	print('dataset size: '+str(len(data)))
	random.shuffle(data)

	# save FIDs after shuffling so that the order is preserved
	row_fids = [row['FID'] for row in data]

	c_data = clean_data( data )
	b_data = bucketize_data( c_data )
	data = b_data

	p = get_mitigation_params()
	p['var_map_fid'] = var_map_fid
	p = prep_params( p )
	
	p['data'] = data

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

	if save: format_save_data(p,data,cols,out_fid=out_fid)

	return p

