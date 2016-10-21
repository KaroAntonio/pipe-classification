import numpy as np
from scipy.optimize import minimize
from fuzzylogic import *
import random 
import json
try:
	import arff
except:
	pass

def params_to_vec( params ):
	'''
	return the bounds of the params as a 1d vec (numpy)
	ignore 0s and infs, they are assumed to be correct
	'''
	vec = []
	param_vars = list(params.keys())
	param_vars.sort()
	for var in param_vars:
		grades = list(params[var].keys())
		grades.sort()
		for grade in grades:
			vec += [e for e in params[var][grade] if e not in [0, float('inf')]]

	return np.array(vec)

def vec_to_params( vec, params ):
	'''
	reconstruct vec into a params dict with the same structure as params
	'''
	vec = list(vec)
	idx = 0  # vec index
	new_params = {}
	param_vars = list(params.keys()	)
	param_vars.sort()
	for var in param_vars:
		new_params[var] = {}
		grades = list(params[var].keys())
		grades.sort()
		for grade in grades:
			bounds = params[var][grade]
			new_bounds = []
			for b in bounds:
				if b in [0,float('inf')]:
					new_bounds += [b]
				else: 
					new_bounds += [vec[idx]]
					idx += 1
			new_params[var][grade] = new_bounds

	return new_params

def get_cl_map():
	return  {
			1:'Extremely Low',
			2:'Very Low',
			3:'Moderately Low',
			4:'Medium',
			5:'Moderately High',
			6:'Very High',
			7:'Extremely High'
			}

def condition_loss_old(vec, params, data):
	
	params = vec_to_params(list(vec), params)

	cl_map = get_cl_map()
	cl_map_inv = {cl_map[k]:k for k in cl_map}

	diffs = []
	for row in data:
		rl = float(row['Remaining Service Life '])
		tb = float(row['No. of Breaks'] )
		rb = float(row['Recent Number of Break']  )
		mi = float(row['Maintenance Index'])
		cl = row['TotalCondition Level']
		x = {'rl':rl,'tb':tb,'rb':rb,'mi':mi}
		out = condition_model(params, x)
		diffs += [abs(out - cl_map_inv[cl])]

	return sum(diffs)/len(diffs) 

def get_loss_old(f, params, data):
	def obj_func(x):
		return f(x, params, data)
	return obj_func

def vec_model_loss(x, p):
	p['params'] = vec_to_params(list(x), p['params'])
	return model_loss( p )

def get_loss_f(p):
	def f(x):
		return vec_model_loss(x, p)
	return f 

def set_arff_nominal( path, nominals ):
	# Cleanse .arff
	f = open(path)
	lines = f.readlines()
	f.close()
	f = open(path, 'w')
	for line in lines:
		# NOMINAL correct class attributes to nominal
		if line[0] == '@' and "{" not in line:
			parts = line.split()
			for c in nominals:
				if c in parts:
					cs = [str(e).replace(' ','_') for e in nominals[c]]
					parts[2] = "{"+", ".join(cs) + "}"
			line = " ".join(parts)+"\n"

		f.write(line)
	f.close()

# ADD back when arff is possible
def data_to_arff( data, arff_fid, var_map=None ):
	if not var_map:headers = list(data[0])
	else: headers = list(var_map)

	arff_data = []
	types = {}
	
	for row in data:
		clean = []
		for h in headers:
			e = row[h]
			e = e.strip()
			e = e.replace(' ','_')
			try:
				e = float(e)
			except:
				pass

			if h not in types:
				types[h] = type(e)

			if type(e) != types[h]:
				if type(e) == str and types[h] == float:
					e = 0.0
				elif type(e) == float and types[h] == str:
					e = str(e)
				else:
					print(type(e),types[h])

			clean += [e]
		arff_data += [clean]

	headers = [e.replace(' ','_') for e in headers]
	arff.dump( arff_fid, arff_data, names=headers)

def optimize_model(p, x0,n=1):

	#data = load_data( 'data/csv/all_data.csv' )
	data = p['data']
	loss = get_loss_f(p)
	for i in range(n):
		print('loss: {}'.format(loss(x0)))
		res = minimize(loss, x0, method='BFGS', 
			options={'disp': True})
		x0 = res.x

	return np.array([round(e,2) for e in res.x])
	return res.x

def rand_update_params(params0):
	return ((np.random.random(len(params0)) *0.3) - 0.15) + params0

def run_optimization(p):
	# load best params
	best_params_vec = params_to_vec( p['params'] )
	loss = get_loss_f(p)
	best_loss = loss(best_params_vec)
	print('best loss: {}'.format(best_loss))

	# randomly update best params and optimize
	rand_params_vec = rand_update_params(best_params_vec)
	optimized = optimize_model(p, rand_params_vec, 2 )
	loss0 = loss(optimized)

	if loss0 < best_loss: best_params_vec = optimized

	model_fid = 'models/condition_model_opt.csv'
	save_model_params(model_fid, vec_to_params(best_params_vec, p['params']))
	
if __name__ == '__main__':

	data_fid = 'data/csv/all_data.csv'
	data = load_data( data_fid )
	random.shuffle(data)

	data_to_arff(data, 'data/arff/all_data.arff', var_map=None )	

	nominals = {}
	nominals['PCSDesc'] = json.load(open('val_maps/condition_exp_map.json', 'r'))
	set_arff_nominal('data/arff/all_data.arff', nominals)

	#p = get_condition_params()
	p = get_criticality_params()

	p['data']=data

	#update fid with data
	p['var_map'] = json.load(open(p['var_map'], 'r'))
	#p['val_map'] = json.load(open(p['val_map'], 'r'))
	p['params'] = load_model_params(p['params'] )

	run_optimization(p)

