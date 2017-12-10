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

	print('ONLY USING FIRST 10 COMBS')
	combs = all_combinations(vars)
	#combs = all_combinations(vars)

	best_comb = None
	best_acc = 0

	for c in combs:	
		p['var_map'] = {0:v for v in  c}
		p['var_map']['out'] = out_var
		acc = assess_acc_n(p, data, 1)
		print('{:1.3f} => {}'.format(acc, c))
		if acc > best_acc:
			best_acc = acc
			best_comb = c

	p['var_map'] = var_map	

	return acc, best_comb

def save_var_map(var_map, var_map_name):
	with open('var_maps/'+var_map_name,'w') as f:
		json.dump(var_map, f)
	
if __name__ == '__main__':
        DATA_FID = 'data/csv/all_data.csv'

        data = load_data( DATA_FID )

	c_data = clean_data( data )
	b_data = bucketize_data( c_data )
	data = b_data

        p = get_mitigation_params()

        # to prep params, data
	p = prep_params( p )	

	acc, best_vars = determine_best_vars(p, data)	
	best_var_map = {0:v for v in best_vars}
	best_var_map['out'] = p['var_map']['out']
	save_var_map(best_var_map,'best_nb.json')
	
