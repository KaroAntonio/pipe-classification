import NaiveBayes
from run_naive_bayes import *
from fuzzylogic import *
import json
import sys
import copy

# Evaluate the features of a trained naive bayes model
# STEPS
# build combinatorial test data (containing rows of ~all~ combinations of features
# train model
# get variance of outputs

def get_attr_vals(data):
	''' 
	return the list of values that each  attr can take in dict:
	{attr:[val1,val2,...],...}
	'''
	attr_vals = {attr:[] for attr in data[0]}
	for row in data:
		for attr in row:
			attr_vals[attr] += [row[attr]]

	# keep only unique vals
	for attr in attr_vals.keys():
		attr_vals[attr] = sorted(list(set(attr_vals[attr])))

	return attr_vals

def build_predict_data(attr_vals, n=1000):
	'''
	build a data set that is composed of a n sample all combinations of attr_vals
		 the total number of combs might be really large
	'''
	pred_data = []
	for i in range(n):
		pred_data += [{attr:random.choice(attr_vals[attr]) for attr in attr_vals}]
	
	return pred_data

def std_dev(data):
	''' data is a 1d list '''
	average = float(sum(data)) / len(data)
	return (sum((value - average) ** 2.0 for value in data) / len(data))** 0.5

def only_relevant_attrs(data, attrs):
	'''
	method to filter out all but the relevent attrs
	'''
	return [{attr:row[attr] for attr in attrs} for row in data]

def attr_std_devs(model, attr_vals):
	''' 
	given a trained model, 
	determine the variance of the avgs of the preds of all vals of an attr
	
	'''
	
	predict_data = build_predict_data(attr_vals)

	# pred {attr: {val:[preds]}}
	predictions = {}

	for row in predict_data:
		row['attributes'] = {e:row[e] for e in row if e != 'model_target'} .copy()
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

		# Record pred
		for attr in row['attributes']:
			if attr not in predictions:
				predictions[attr] = {}
			if row['attributes'][attr] not in predictions[attr]:
				predictions[attr][row['attributes'][attr]] = []

			predictions[attr][row['attributes'][attr]] += [out]

	pred_std_devs = {attr:std_dev(predictions[attr][row[attr]]) for attr in attr_vals if attr != 'model_target'}
	return pred_std_devs

def train_assess_nb(p,data):

	data = bucketize_data_attrs(data)
	attrs = p['var_map'].values()

	train_data = only_relevant_attrs(data,attrs+['model_target'])
	attr_vals = get_attr_vals(train_data)

	predict_data = build_predict_data(attr_vals)

	model = build_train_model(p,data,train_data)

	return attr_std_devs(model, attr_vals)

if __name__ == '__main__':
	DATA_FID = 'data/csv/all_data.csv'

	data = load_data( DATA_FID )
	p = get_mitigation_params()	

	# to prep params, data
	naive_bayes_accuracy(p,data)

	std_devs = train_assess_nb(p,data)
	print(std_devs)

