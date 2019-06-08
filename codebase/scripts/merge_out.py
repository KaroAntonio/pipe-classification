'''
script to merge the out files according to FID
'''

from utils import *

script_map = {
		'nb_out_data.csv':'naive_bayes',
		'fuzzy_out_data.csv':'fuzzy'
		}

merged_fid = 'out/merged_out_data.csv'

data = {}
for fid,model_name in script_map.items():
	for row in load_data('out/'+fid):
		row_fid = row['FID']
		if row['FID'] not in data:
			data[row_fid] = {}
		del row['FID']
		for key,val in row.items():
			data[row_fid][model_name + '_' +key] = val

# validate, reinsert FID
s = len(data[data.keys()[0]])
for row_fid in data:
	if len(data[row_fid]) != s:
			print('Num Values Mismatch')
	data[row_fid]['FID'] = row_fid

save_data(merged_fid,data.values())






	
	
