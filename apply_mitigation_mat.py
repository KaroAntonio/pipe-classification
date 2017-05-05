from fuzzylogic import *

mat_fid = 'data/csv/mitigation_matrix.csv'
data_fid = 'data/csv/all_data.csv'
data = load_data( data_fid )
mat_data = load_data( mat_fid )

# get headers in correct order
f = open(data_fid) 
headers = list(set([ e.strip('\n') for e in f.readline().split(',')]))
#headers = [ e.strip() for e in f.readline().split(',')]
csv_headers = list(data[0]) 
#csv_headers = [e.strip() for e in csv_headers]
#headers = [ e for e in headers if e.strip()]
diff1 = [ e for e in headers if e not in csv_headers]
diff2 = [ e for e in csv_headers if e not in headers]

print(diff1+diff2, len(headers), len(csv_headers))

input_map = {
	'Condition':'Condition_level',
	'performance':'Performance_Level',
	'Criticality':'Criticality_Level'
}

mit_k = 'Mitigation_Matrix_Score'

for row in data:
	for mat_row in mat_data:
		is_match = True
		for mat_k,data_k in input_map.items():
			if row[data_k] != mat_row[mat_k]:
				is_match = False
		if is_match:
			row[mit_k] = mat_row['Mitigation']

save_data('out/all_data_mitigation.csv',data, [mit_k]+headers)

