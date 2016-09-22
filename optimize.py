import numpy as np
import arff
from scipy.optimize import minimize
from fuzzylogic import *

def params_to_vec( params ):
	'''
	return the bounds of the params as a 1d vec (numpy)
	ignore 0s and infs, they are assumed to be correct
	'''
	vec = []
	param_vars = params.keys()	
	param_vars.sort()
	for var in param_vars:
		grades = params[var].keys()
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
	param_vars = params.keys()	
	param_vars.sort()
	for var in param_vars:
		new_params[var] = {}
		grades = params[var].keys()
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

	cl_map = {
			1:'Extremely Low',
			2:'Very Low',
			3:'Moderately Low',
			4:'Medium',
			5:'Moderately High',
			6:'Very High',
			7:'Extremely High'
			}

	return cl_map

def condition_loss(vec, params, data):
	
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

def get_loss(f, params, data):
	def obj_func(x):
		return f(x, params, data)
	return obj_func

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
						cs = [str(e) for e in nominals[c]]
						parts[2] = "{"+", ".join(cs) + "}"
				line = " ".join(parts)+"\n"

			f.write(line)
        f.close()

def data_to_arff( data, arff_fid ):

	cl_map = get_cl_map()
	cl_map_inv = {cl_map[k]:k for k in cl_map}

	arff_data = []
	headers = ['rl','tb','rb','mi','cl']
	for row in data:
		rl = float(row['Remaining Service Life '])
		tb = float(row['No. of Breaks'] )
		rb = float(row['Recent Number of Break']  )
		mi = float(row['Maintenance Index'])
		cl = cl_map_inv[row['TotalCondition Level']]
		arff_data += [[rl,tb,rb,mi,cl]]
	arff.dump( arff_fid, arff_data, names=headers)


def optimize_condition_model(params0,n=1):

	data = load_data( 'data/csv/condition_data_2014.csv' )
	loss = get_loss(condition_loss, params0, data)
	x0 = params_to_vec(params0)
	for i in range(n):
		print('loss: {}'.format(loss(x0)))
		res = minimize(loss, x0, method='BFGS', 
			options={'disp': True})
		x0 = res.x

	return np.array([round(e,2) for e in res.x])
	return res.x
	
if __name__ == '__main__':
	data = load_data( 'data/csv/condition_data_2014.csv' )
	data_to_arff(data, 'data/csv/condition_data_2014.arff' )	
	set_arff_nominal('data/csv/condition_data_2014.arff', {'cl':get_cl_map()})

	# load best params
	best_params = load_model_params( 'models/condition_model_opt.csv' )
	best_x = params_to_vec( best_params )
	loss = get_loss(condition_loss, best_params, data)
	best_loss = loss(best_x)
	print('best loss: {}'.format(best_loss))

	# randomly update best params and optimize
	params0 = load_model_params( 'models/condition_model_opt.csv' )
	rand_vec = (np.random.random(len(params_to_vec(params0))) *0.3) - 0.15
	rand_params = vec_to_params( best_x + rand_vec, params0 )
	optimized = optimize_condition_model( rand_params, 2 )
	loss0 = loss(optimized)

	if loss0 < best_loss: best_x = optimized

	model_fid = 'models/condition_model_opt.csv'
	save_model_params(model_fid, vec_to_params(best_x, params0))
	
