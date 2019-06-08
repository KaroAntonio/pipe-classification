import csv
import json

def parse_val(val):
	if type(val) == str:
		val = val.strip()
	try:
		return float(val)
	except:
		return val

def load_data( fid ):
        with open( fid, 'r' ) as csvfile:
                data_reader = csv.DictReader( csvfile )
                rows = []
                for row in data_reader:
                        rows += [row]
        return rows

def save_data( fid, data, fieldnames=None ):
        if not fieldnames:
                fieldnames = list(data[0])

        with open( fid, 'w' ) as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for row in data:
                        writer.writerow(row)

def format_save_data(p,data, cols=None, out_fid='out/nb_out_data.csv'):
        if not cols:
                cols = [
                                p['var_map']['out'],
                                p['model_name']+'_pred'
                                ]

        out_data = [{k:v for  k,v in row.items() if k in cols} for row in  data]
        save_data(out_fid,out_data,cols)

def build_str_maps( attr_vals, attr_types ):

	# a map of a str value to a float	
	str_maps = { attr:{} for attr in attr_vals }	
	
	for attr,t in attr_types.items():
		if t in [str, 'mixed']:
			str_map = {val:float(i) for i,val in enumerate(set(attr_vals[attr])) }
			str_maps[attr] = str_map 
	return str_maps

def assess_attr(data):
	
	attr_vals = { attr:[] for attr in data[0].keys() }	
	attr_types = { attr:type(parse_val(v)) for attr,v in data[0].items() }

	# detect attr types
	for row in data:
		for attr,v in row.items():
			val = parse_val(v)
			attr_vals[attr] += [val]
			
			if type(val) != attr_types[attr]:
				attr_types[attr] = 'mixed'

	attr_vals = {attr:list(set(vals)) for attr, vals in attr_vals.items() }

	return attr_vals, attr_types

def map_vals( data, str_map ):
	c_data = []

	for row in data:
		clean_row = {}
		for attr,v in row.items():
			val = parse_val(v)
			if type(val) in [str,'mixed']:
				clean_row[attr] = str_map[attr][val]
			else:
				clean_row[attr] = val
		c_data.append(clean_row)

	return c_data

def all_vals_float( data ):
	# return true iff all type( vals ) == float 
	for row in data:
		for v in row.values():
			if type(v) != float: return False
	return True 

def clean_data( data ):
	# turn all elements into floats
	# NOTE: Treating mixed attrs as str

	attr_vals, attr_types = assess_attr( data )

	str_map = build_str_maps(attr_vals, attr_types)

	return map_vals( data, str_map )

def get_attr_stats( attr_vals ):
	
        attr_stats = { attr:{}  for attr in attr_vals }

        for attr, vals in attr_vals.items():
                attr_stats[attr]['n'] = float(len(vals))
                attr_stats[attr]['max'] = max(vals)
                attr_stats[attr]['min'] = min(vals)
                attr_stats[attr]['range'] = attr_stats[attr]['max'] - attr_stats[attr]['min']

	return attr_stats

def bucketize_data(data, n=10):
        '''
        for each attribute, adjust it to fall into n buckets
                if there are more than n unique vals
        '''

        b_data = []

        attr_vals, attr_types = assess_attr( data )
	attr_stats = get_attr_stats( attr_vals )

        for row in data:
                b_row = {}
                for attr,val in row.items():
                        if attr_stats[attr]['n'] > n:
                                val_min = attr_stats[attr]['min']
                                val_range = attr_stats[attr]['range']
                                bucket_size = val_range / float(n)
                                b_val = round(((val-val_min)/val_range)*n)*bucket_size
                                b_row[attr] = b_val
			else:
				b_row[attr] = val
                b_data.append(b_row)

        return b_data

def get_attr_val_counts(data, attr):
	'''
	return a dict of the counts for each of the attrs vals
	'''
	attr_val_counts = {}
	for row in data:
		val = row[attr]	 
		if val not in attr_val_counts:
			attr_val_counts[val] = 0
		attr_val_counts[val] += 1 
	return attr_val_counts

def prompt_for_valid_filename(prompt):
		
	while(1):
		fid = raw_input(prompt).strip()
		try:
			f = open(fid)
			f.close()
			return fid
		except:
			print('{} is not a file.'.format(fid))
		

def load_prep_data( data_fid ):
	'''
	load, clean, bucketize data
	'''
	
	data = load_data( data_fid )

        c_data = clean_data( data )
        b_data = bucketize_data( c_data )
	return b_data

if __name__ == '__main__':

	TRAINING_FID = 'data/csv/all_data.csv'
	data = load_data( TRAINING_FID )

	c_data = clean_data( data )

        attr_vals, attr_types = assess_attr( c_data )
	attr_stats = get_attr_stats( attr_vals )

	b_data = bucketize_data( c_data )

        b_attr_vals, b_attr_types = assess_attr( b_data )
	b_attr_stats = get_attr_stats( b_attr_vals )

	all_vals_float(c_data)


