from random import random as rand
import csv

def load_data( fid ):
	with open( fid, 'r' ) as csvfile:
		data_reader = csv.DictReader( csvfile )
		rows = []
		for row in data_reader:
			rows += [row]
	return rows	

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
			# print( x, p1,p2 , line_y_intersect(x, p1, p2)  )
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
	degrees = {l:degrees[l]/total for l in degrees};

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

def test_condition_model(fid, params_fid):
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
		tb = float(row['No. of\nBreaks'] )
		rb = float(row['Recent Number of Break']  )
		mi = float(row['Maintenance Index'])
		cl = row['TotalCondition Level']
		x = {'rl':rl,'tb':tb,'rb':rb,'mi':mi}
		out = condition_model(params, x)
		cl_out  = cl_map[round(out)]
		diff = cl_map_inv[cl] - cl_map_inv[cl_out]

		# TRACK RESULTS
		count += 1
		if diff not in diffs: diffs[diff] = 0
		else: diffs[diff] += 1
		print(str(diff) + ', actual: '+cl+', ''model: '+ cl_out)

	diff_ratio = { k:diffs[k]/float(count) for k in diffs }	
	print( diff_ratio )

def save_model_params( fid, params ):
	'''
	save model params to file fid
	'''
	f = open(fid, 'w')
	f.write(",".join(['var','grade','bounds']) + '\n')
	for var in params:
		for grade in params[var]:
			bounds = params[var][grade]
			f.write(",".join([str(e) for e in [var]+[grade]+bounds])+"\n")
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
	
	try:
		params = {}
		for line in f:
			parts = line.split(',')
			parts = [e.strip() for e in parts]
			if 'var' not in parts:
				if parts[0] not in params:
					params[parts[0]] = {}
				params[parts[0]][int(parts[1])] = [float(e) for e in parts[2:]]
	except: 
		print('Parse Error')
		return None
	f.close()
	return params

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
	for var in x:
		memberships[var] = membership(x[var], params[var])

	out = sum(memberships.values()) * 1./6

	return membership( out, params['out'])


def condition_model_static(rl, tb, rb, mi ):
	'''
	rl: Remaining Life
	tb: Total Breaks
	rb: Recent Number of Breaks
	mi: Maintenance Index	
	return an integer condition rating
	'''

	# NOTE what are the mi bounds? 
	# desc is in range 0 - 5%, bounds is in (0,0.06)...?
	# should the bounds at the top end always go to infinity ...?

	rl_bounds = {
			0: [45,53,54, float('inf')],
			5: [25,33,48,55],
			15: [10,17,28,35],
			20: [0,0,12,20] 
			}

	tb_bounds = {
			0: [0,0,1,2],
			5: [0,2,4.5, 6],
			15: [4,5.5,8,10],
			20: [8,9,10,float('inf')]
			}

	rb_bounds = {
			0: [0,0,0.5,1],
			5: [0.5,1,2.5,3],
			15: [2.5,3,4.5,5],
			20: [4.5,5,6,float('inf')]
			}
	
	mi_bounds = {
			5: [0,0,0.01,0.02],
			10: [0.01,0.02,0.04,0.05],
			15: [0.04,0.045,0.06,float('inf')]
			}

	out_bounds = {
			1: [0,0,1.75],
			2: [0,1.75,3.25],
			3: [1.75,3.25,5],
			4: [3.25,5,6.5],
			5: [5,6.5,8],
			6: [6.5,8,10],
			7: [8,10,float('inf')]
			}
	
	rl_out = membership( rl, rl_bounds )
	tb_out = membership( tb, tb_bounds )
	rb_out = membership( rb, rb_bounds )
	mi_out = membership( mi, mi_bounds )
	
	# print( rl_out, tb_out, rb_out, mi_out )
	
	# Input to the condition model bound is the sum of the prior outputs
	out = sum([rl_out, tb_out, rb_out, mi_out]) * 1./6

	return membership( out, out_bounds )

def performance_model( hc, qy, cs, pd, pt, land_use_type ):
	'''
	pt: Pipe Type
	hc: Hydraulic Capacity
	qy: Quality
	cs: Conformance to Standard
	pd: Pipe Diameter (mm)
	return an integer performance grade
	'''
	if pd >= 600:	
		hc_bounds = {
				0:[0,0,1,1.75],
				15:[1,1.5,2.25,2.75],
				30:[2,2.5,3.5,float('inf')]
				}
	else:
		hc_bounds = {
				0:[0,0,1,2.25],
				15:[1,2,4,5.5],
				30:[4,5,6,float('inf')]
				}

	
	qy_bounds = {
			0:[0,0,0.5],
			15:[14.5,15,15.5]
			}

	inv_land_use_cats = [
				'Industry',
				'Schools'
			]
	if pd < 19: cs_out = 10
	elif pd < 100 and pt == "Copper": cs_out = 15
	elif pd < 300 and land_use_type in inv_land_use_cats : cs_out = 15
	else: cs_out = 0

	out_bounds = {
			1: [0,0,1.75],
			2: [0,1.75,3.25],
			3: [1.75,3.25,5],
			4: [3.25,5,6.5],
			5: [5,6.5,8],
			6: [6.5,8,10],
			7: [8,10,float('inf')]
			}

	hc_out = membership( hc, hc_bounds )
	qy_out = membership( qy, qy_bounds )
	cs_out = membership( cs, cs_bounds )

	out = sum([hc_out, qy_out, cs_out]) * 2./9 # 1/6 * 4/3
	
	return membership( out, out_bounds )
	

def criticality_model( pd, es, ac ):
	'''
	pd: Pipe Diameter
	es: Environmentally Sensitive
	ac: Accessibility
	'''
	pd_bounds = {
			0:[0,0,300,500],
			10:[300,400,750,850],
			15:[650,750,float('inf')],
			}

	es_bounds = {
			0:[0,0,10],
			15:[6,15,float('inf')],
			}

	ac_bounds = {
			0:[0,0,6],
			10:[4,8,12],
			15:[10,16, float('inf')],
			}

	out_bounds = {
			2: [0,1.75,3.25],
			3: [1.75,3.25,5],
			4: [3.25,5,6.5],
			5: [5,6.5,8],
			6: [6.5,8,float('inf')],
			}

	pd_out = membership( pd, pd_bounds )
	es_out = membership( es, es_bounds )
	ac_out = membership( ac, ac_bounds )

	out = sum([pd_out, es_out, ac_out]) * 2./9 # 1/6 * 4/3
	
	return membership( out, out_bounds )

if __name__ == "__main__":
	params_fid = 'models/condition_model_opt.csv'
	test_condition_model('data/condition_data_2014.csv', params_fid )
	

