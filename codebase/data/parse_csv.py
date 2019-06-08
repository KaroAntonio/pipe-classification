import csv, re, datetime

# map of file id's to the from/to format for each file
fids = {
		#'161301_WS_Replacement_Mississauga.csv':['From','To'],
	#'161310_WMN Replacement Program_Mississauga_Updated on Dec 07, 2015.csv':['From','To'],
	'Transmission Data.csv':['FRM_NODE','TO_NODE'],
	'Watermain Break Records Copper.csv':['FROM NODE','TO NODE'],
	'Watermain Break Records Wolfedale.csv':['FROM NODE','TO NODE'],
	'Watermain Break Records.csv':['FROM NODE','TO NODE'],
	'condition_data_2014.csv':['FROM NODE','TO NODE'],
	}

dir_path = 'csv/'

data = {}
pipes = {}

cross_file_hits = 0

for fid in fids:
	print(fid)
	f = open(dir_path+fid, 'r')
	reader = csv.DictReader(f)
	f_data = []
	for row in reader:
		# get pipe as defined by from node, to node
		from_node = re.sub("[^0-9]", "", row[fids[fid][0]])
		to_node = re.sub("[^0-9]", "", row[fids[fid][1]])
		pipe_id = from_node, to_node 
	
		'''
		if pipe_id == ('',''):
			# EXIT AT NULL PIPE
			print(pipe_id)
			print( fid )
			print( row )
			exit()
		'''

		# skip non null rows
		if not all([e=='' for e in row]) and pipe_id !=  ('',''):
			f_data += [row]

			# PROCESS BREAK RECORDS
			if "Break Records" in fid:
				if 'No. of Breaks' in row:
					no_breaks = row['No. of Breaks']
				elif 'No. Breaks' in row:
					no_breaks = row['No. Breaks']
				else:
					print('no break record')
					print(row)
				if 'W/O Date' in row:
					this_year = datetime.date.today().year
					record_year = int(row['W/O Date'].split('/')[2])
					age = this_year-record_year
					if age > 40:
						print(age)

				if 'status' in row:
					print('AHHHH')
					exit()
				row['pipe_status'] = 'replaced' if no_breaks > 5 else 'repaired'

			if pipe_id not in pipes:
				pipes[pipe_id] = [fid]
			else:
				if fid not in pipes[pipe_id]:
					cross_file_hits += 1
					pipes[pipe_id] += [fid]

	data[fid] = f_data 

n_greater_one = 0
fid_counts = {}
for p in pipes:
	if len(pipes[p]) > 1:
		n_greater_one += 1
		if ' '.join(pipes[p]) not in fid_counts:
			fid_counts[' '.join(pipes[p])] = 0
		fid_counts[' '.join(pipes[p])]  += 1
		
print( 'n greater one ' + str(n_greater_one)) 
for fid in fid_counts:
	print(fid, fid_counts[fid])
