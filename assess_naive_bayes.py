import ml_models.NaiveBayes
from naive_bayes_utils import *
import json
import sys
import copy
import itertools

'''
	assess NB V2 
	strat: run NB with each of the vars unincluded to assess 
	should probably do 10 runs of each to get stat reliability
'''


def all_combinations(l):
	combs = []
	for i in range(1,len(l)+1):
		combs += itertools.combinations(l,i)
	return combs

def assess_acc_n(p, data, n):
	runs = []
	for i in range(n):
		runs += [label_predictions(p,data)]
	return sum(runs)/float(n)


def determine_best_vars(p, data):
	# assume params are prepped
	# returns acc, best_vars
	
	out_var = p['var_map']['out']	
	var_map = p['var_map']
	vars = [v for k,v in p['var_map'].items() if k != 'out']

	combs = all_combinations(vars)

	best_comb = None
	best_acc = 0

	for c in combs:	
		p['var_map'] = {i:e for i,e in enumerate(c)}
		p['var_map']['out'] = out_var
		acc = assess_acc_n(p, data, 1)
		print('{:1.3f} => {}'.format(acc, c))
		if acc > best_acc:
			best_acc = acc
			best_comb = c

	p['var_map'] = var_map	

	return acc, best_comb

def normalize_weights(var_weights):
	vars = var_weights.values()
	max_weight = max(vars)
	min_weight = min(vars)
	
	for var, weight in var_weights.items():
		var_weights[var] = (weight - min_weight)  / (max_weight - min_weight)
	return var_weights
	

def determine_var_weights(p, data):
	'''
	calc the acc for model with each of the vars dropped
	'''

	out_var = p['var_map']['out']
        var_map = p['var_map']
        vars = [v for k,v in p['var_map'].items() if k != 'out']

	var_weights = {}	
	for var in vars:
		var_map = { i:e for i,e in enumerate(vars) if e != var }
		var_map['out'] = out_var
		p['var_map'] = var_map

		acc = assess_acc_n(p, data, 1)
		var_weights[var] = round(acc, 4)
		print('{:1.3f} => {}'.format(acc, var_map))

	var_weights = normalize_weights(var_weights)
	var_weights = {var:round(1-weight,4) for var, weight in var_weights.items()}
	return var_weights 
	
def save_var_map(var_map, var_map_name):
	with open('var_maps/'+var_map_name,'w') as f:
		json.dump(var_map, f)

def determine_save_best_vars(p, data):
	acc, best_vars = determine_best_vars(p, data)	
	best_var_map = {i:v for i,v in enumerate(best_vars)}
	best_var_map['out'] = p['var_map']['out']
	save_var_map(best_var_map,'best_nb.json')
	
if __name__ == '__main__':
        DATA_FID = 'data/csv/all_data.csv'
	VAR_MAP_FID = 'var_maps/mitigation_var_map.json'

        data = load_prep_data( DATA_FID )
	p = get_mitigation_params()
	p['var_map_fid'] = VAR_MAP_FID
	p = prep_params( p )	
	
	#determine_save_best_vars(p,data)	

	var_weights = determine_var_weights(p, data)
	print var_weights
	save_data('var_weights.csv', [var_weights])
	
