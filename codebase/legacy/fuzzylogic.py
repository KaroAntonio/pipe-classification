from random import random as rand
import random
import csv,json
#from params import *

def load_data( fid ):
	with open( fid, 'r' ) as csvfile:
		data_reader = csv.DictReader( csvfile )
		rows = []
		for row in data_reader:
			rows += [row]
	return rows	

def gen_col( data, name, b):
	'''
	generate a column in the data s.t. in the range 0,b
	where b, are strings that evaluate under
	eval(a)
	'''
	for row in data:
		row[name] = int(rand() * eval(b))

def save_data( fid, data, fieldnames=None ):
	if not fieldnames:
		fieldnames = list(data[0])

	with open( fid, 'w' ) as csvfile:
		writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
		writer.writeheader()
		for row in data:
			writer.writerow(row)

def line_y_intersect(x, p1, p2):
	'''
	p1, p2: (px,py)
	return y for y = mx+b where y=mx+b is the line defined by p1->p2
	'''
	# slope = dy / dx
	m = (float(p2[1]) - p1[1]) / (p2[0] - p1[0])
	
	# b = y - mx
	b = p1[1] - m*p1[0]

	return m*x+b

def seg_deg_membership(i, x, bounds):
	'''
	i: index of segment
	x: x coordinate 
	bounds: as below
	return degree of membership, for this segmont of the bounds
	'''
	# IF start segment
	if i == 0:
		if bounds[i] == bounds[i+1]:
			return 1
		else:
			# UPDATE WITH LINE Y FUNC
			p1 = (bounds[i], 0)
			p2 = (bounds[i+1],1)
			return line_y_intersect(x, p1, p2) 

	# IF plateau segment
	if i == 1 and len(bounds) == 4:
		return 1

	# IF end segment
	if (i == 1 and len(bounds) == 3) or i == 2:
		p1 = (bounds[i], 1)
		p2 = (bounds[i+1],0)
		return line_y_intersect(x, p1, p2) 

def bounds_membership(x, bounds ):
	'''
	x: float in  (0,1)
	bounds: triangle or trapezoid
	return the membership ratio of x  within bounds
	'''
	
	for i in range(len(bounds)-1):
		# if x is within bound segment
		if x >= bounds[i] and x <= bounds[i+1]:
			return seg_deg_membership(i, x, bounds)

def membership_degrees( x, memberships ):
	'''	
	return the membership degrees for all the relevant bounds
	'''
	degrees = {}
	for score in memberships:
		b = memberships[score]
		if x >= b[0] and x <= b[-1]:
			degrees[score] = bounds_membership(x, b)
	
	return degrees

def validate_memberships( memberships ):

	# bounds must have either 3 or 4 elements, 
	# and must be in non-descending order
	for b in memberships.values():
		if len(b) != 3 and len(b) != 4:
			raise Exception("Invalid Membership Bounds.")

		test = b[:]
		test.sort()
		if test != b:
			raise Exception("Bounds must be in Non-Decreasing Order.")

def choose_membership( degrees ):
	'''
	choose membership incrementally
	choose membership with the associated probability
	'''
	c=random.uniform(0,1)
	for label in degrees:
		if c < degrees[label]: return label

		else: c -= degrees[label]

	# if, somehow, no label wash chosen, try again
	raise Exception('Should have chosen')
	return choose_membership( degrees )

def weighted_sum( degrees ):
	'''
	return 
	'''
	ws = 0.0

	for k in degrees:
		if type(k) != int:
			raise Exception("Labels not Integers.")
		ws += float(k)*degrees[k]

	return ws

def membership( x, memberships ):
	'''
	memberships: {'label':bounds}
	bounds may be a triangle or a trapezoid
	bounds: [0, 5, 10, 20] OR [0, 4, 8]
	TODO must bounds be consistent? 
	aka at each point must the memberships sum to 1?
	return the membership for x according to the trapezoids defined by classes
	'''
		
	# VALIDATE 
	# validate_memberships(memberships)

	# SORT
	# ensure memberships are in non-decreasing order
	for label in memberships:
		bounds = memberships[label]
		bounds.sort()
		memberships[label] = bounds

	# FIND degree of membership to each bound
	degrees = membership_degrees(x, memberships)
	# NORMALIZE degrees to probabilities
	total = sum([degrees[k] for k in degrees])
	if total: degrees = {l:degrees[l]/total for l in degrees};

	#return choose_membership( degrees )
	return weighted_sum( degrees )

def gen_rand_data( n, ranges ):
	data = []
	for i in range(n):
		data += [[int(rand() * ranges[i]) for i in range(len(ranges))]]
	return data

def average_missing( data ):
	'''
	data: list of dicts [{'a':1,'b':2},{'a':2,'b':4}, ...]
	missing values are None types
	fill in missing values with the average of the column
	'''
	pass

def test_criticality_model(fid, params_fid, out_fid=None):

	data = load_data( fid )
	params = load_model_params( params_fid )
	diffs = {}
	count = 0
	for row in data:
		x = { k:row[v] for k,v in var_map.items() if k != 'out' }
		out = criticality_model(params, x)
		print(out*100, row[var_map['out']])

def test_performance_model(fid, params_fid, out_fid=None):
	
	data = load_data( fid )
	params = load_model_params( params_fid )

	'''
	var_map = {
		'pt': Pipe Type
		'hc': Hydraulic Capacity
		'qy': Quality
		'cs': Conformance to Standard
		'pd': Pipe Diameter (mm)
		}
		'''

def performance_exp_out(p,row):

	out = row[p['var_map']['out']]
	if (type(out) == str):
		val = out.strip()
	elif type(out) == int or type(out) == float:
		val = out 
	else:
		raise Exception('WHAT ARE UUU')

	try:
		return float(val)
	except:
		return val

def condition_calc_out(row):
		# the calculated output
		# true out in [0,60]
		out_cols = [
				row['BRK_SCR'],
				row['RSL_SCR'],
				int(row['BRKS_FVYRS_SCR'])*3,
				row['MI_SCR']
				]
		return sum(int(e) for e in out_cols)/60.

def criticality_exp_out(p,row):
	
	out = row[p['var_map']['out']]
	if (type(out) == str):
		val = out.strip()
	elif type(out) == int or type(out) == float:
		val = out 
	else:
		raise Exception('WHAT ARE UUU')
	try:
		ret = float(val)
	except:
		if val == '`':
			ret=0
		else:
			#print(val)
			ret = 0
	return ret

def condition_exp_out(p,row):
	k = row[p['var_map']['out']]
	if type(k) == str: 
		k.strip()
		if k.isdigit():
			return int(k)
	if type(k) == float:
		return k
	return p['val_map'][k]/20.

def model_loss(p,n=50):
	# data, params, model, var_map, out_func
	# unpack params to vars
	data = p['data']
	var_map = p['var_map']
	model = p['model']
	params = p['params']
	out_f = p['out_f']

	diffs = []
	if not n: n = len(data)
	i0 = int(rand() * (len(data)-n-1))
	for row in data[i0:i0+n]:
		x = { k:row[v] for k,v in var_map.items() if k != 'out' }
		m_out = model(params, x) 
		out = out_f(p,row)
		diffs += [abs(out-m_out)]
	return sum(diffs)/len(diffs)

def test_condition_model_old(fid, params_fid, out_fid=None):
	'''
	fid: dataset path
	'''
	data = load_data( fid )
	params = load_model_params( params_fid )
	
	cl_map = {
			1:'Extremely Low',
			2:'Very Low',
			3:'Moderately Low',
			4:'Medium',
			5:'Moderately High',
			6:'Very High',
			7:'Extremely High'
			}

	cl_map_inv = {cl_map[k]:k for k in cl_map}

	diffs = {}
	count = 0
	for row in data:
		rl = float(row['Remaining Service Life '])
		tb = float(row['No. of Breaks'] )
		rb = float(row['Recent Number of Break']  )
		mi = float(row['Maintenance Index'])
		cl = row['TotalCondition Level']
		x = {'rl':rl,'tb':tb,'rb':rb,'mi':mi}
		out = condition_model(params, x)
		cl_out  = cl_map[round(out)]
		diff = cl_map_inv[cl] - cl_map_inv[cl_out]
		row['condition_model'] = cl_out

		# TRACK RESULTS
		count += 1
		if diff not in diffs: diffs[diff] = 0
		else: diffs[diff] += 1
		print(str(diff) + ', actual: '+cl+', ''model: '+ cl_out)

	diff_ratio = { k:diffs[k]/float(count) for k in diffs }	
	#print( diff_ratio )
	
	# Save Predictions
	if out_fid == None:
		out_fid = 'out/'+fid.split('/')[-1]

	save_data(out_fid, data)

def save_model_params( fid, params ):
	'''
	save model params to file fid
	'''
	f = open(fid, 'w')
	f.write(",".join(['var','grade','bounds']) + '\n')
	for var in params:
		for grade in params[var]:
			bounds = params[var][grade]
			line = (",".join([str(e) for e in [var]+[grade]+bounds])+"\n")
			f.write(line)
	f.close()

def load_model_params( fid ):
	'''
	load model from file fid
	return None if fid does not exist
	'''
	try:	
		f = open(fid, 'r')
	except:
		return None
	
	params = {}
	for line in f:
		if not line.strip(): 
			print('Err: blank line in params')
			raise Exception
		parts = line.split(',')
		parts = [e.strip() for e in parts]
		if 'var' not in parts:
			if parts[0] not in params:
				params[parts[0]] = {}
			params[parts[0]][int(parts[1])] = [float(e) for e in parts[2:]]

	f.close()
	return params

def performance_model(params, x):
	'''
	TODO
	'''
	raise Exception('NOT IMPLEMENTED')

def criticality_model( params, x ):
	'''
	params: model params
	x: input vars in dict

	pd: Pipe Diameter
	es: Environmental Sensitivity
	ac: Accessibility
	'''
	# For each input variable find membership given bounds
	memberships = {var:membership(float(x[var]),params[var]) for var in x}

	out = sum(memberships.values()) * 2./9
	return membership( out, params['out'])/float(max(params['out'].keys()))

def condition_model( params, x ):
	'''
	params: parameters for bounds and grades of model vars
	x: input vars in the form
		{ 'rl':value, 'tb':value, ... }
	rl: Remaining Life
	tb: Total Breaks
	rb: Recent Number of Breaks
	mi: Maintenance Index	
	return an integer condition rating
	'''
	memberships = {}
	# For each input variable (rl, tb, ...) find membership given bounds
	memberships = {var:membership(float(x[var]),params[var]) for var in x}
	out = sum(memberships.values()) * 1./8

	return membership( out, params['out'])/7.

def gen_brks_cols():
	data_fid = 'data/csv/all_data.csv'
	data= load_data('data/csv/all_data.csv')
	gen_col(data,'TotBrks','200')
	gen_col(data,'BRKS_FVYRS_SCR','row["TotBrks"]')
	save_data('data/csv/all_data_2.csv',data)

def mitigation_model( params, x ):
	return 1

def get_condition_params():
	return {
			'model_name': 	'condition_model',
			'model':		condition_model, 
			'var_map_fid': 		'var_maps/condition_var_map.json',
			#'out_f':		condition_calc_out,
			'out_f':		condition_exp_out,
			'val_map_fid':  	'val_maps/condition_exp_map.json',
			'params_fid':		'models/condition_model_opt.csv'
			}

def get_criticality_params():
	return {
			'model_name': 	'criticality_model',
			'model':		criticality_model, 
			'var_map_fid': 		'var_maps/criticality_var_map.json',
			'out_f':		criticality_exp_out,
			'params_fid':		'models/criticality_model_opt.csv'
			}

def get_performance_params():
	return {
			'model_name': 	'performance_model',
			'model':		performance_model, 
			'var_map_fid': 		'var_maps/performance_var_map.json',
			'out_f':		performance_exp_out,
			'params_fid':		'models/performance_model_opt.csv'
			}

def get_mitigation_params():
    return {
			'out_f':		condition_exp_out,
            'model_name':   'mitigation_model',
            'var_map_fid':      'var_maps/mitigation_var_map.json',
            }

def sample_data(p,data, n_bkts=10):
	'''
	sample data according to it's distribution 
	into n_bkts buckets
	'''
	outs = [p['out_f'](p,row) for row in data]
	max_val = max(outs)
	min_val = min(outs)
	ran = max_val - min_val
	bucket_size = round(ran / float(n_bkts),3)
	out_dist = {(bucket_size * i):0 for i in range(n_bkts+1)}
	for val in outs:
		val = int(val/bucket_size)*bucket_size
		if val in out_dist:
			out_dist[val] += 1
	dist_counts = out_dist.values()	
	dist_counts.sort()

def run_model(data,params):
	# adds a column coresponding to the output of the model to the data

	p = params
	p['data']=data
	
	#update fid with data
	if 'var_map_fid' in p: p['var_map'] = json.load(open(p['var_map_fid'], 'r'))
	if 'val_map_fid' in p: p['val_map'] = json.load(open(p['val_map_fid'], 'r'))
	p['params'] = load_model_params( p['params_fid'] ) 

	#sample = sample_data(params,data)

	loss = model_loss(p,None)
	print(p['model_name']+' loss: ' + str(loss))
	out_data = []
	for row in data:
		out = p['out_f'](p,row)
		#print(out)
		row[p['model_name']] = int(out)

def build_data_frame(p):
	data_frame = {}
	for model_attr,table_attr in p['var_map'].items():
		data_frame[table_attr] = []
		for row in p['data']:
			try:
				data_frame[table_attr] += [float(row[table_attr])]
			except:
				pass

	return data_frame

if __name__ == "__main__":
	data_fid = 'data/csv/all_data.csv'
	data = load_data( data_fid )

	cop = get_condition_params()
	cop['model'] = condition_model
	crp = get_criticality_params()
	#pep = get_perfromance_params()

	run_model(data,cop)
	run_model(data,crp)

	cop_df = build_data_frame(cop)
	crp_df = build_data_frame(crp)

	# choose only certain cols
	cols = ['FID',
			'condition_model',
			crp['var_map']['out'],
			cop['var_map']['out'],
			'criticality_model',
			]

	out_data = [{k:v for  k,v in row.items() if k in cols} for row in  data]
	for row in out_data:
		# normalize
		'''
		try:
			row['PCSDesc']=cop['val_map'][row['PCSDesc']]/20.
		except:
			pass
		try:
			row['CRITICAL_SCR']=float(row['CRITICAL_SCR'])/1000.
		except:
			row['CRITICAL_SCR'] = 0
		'''
	
	save_data('out/fuzzy_out_data.csv',out_data,cols)
	
