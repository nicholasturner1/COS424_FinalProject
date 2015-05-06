import numpy as np

def import_file(filename, y_var = "yr"):

	year_index = {
		'yr' : 0,
		'yr5' : -2,
		'yr10' : -1
	}

	f_obj = open(filename)

	lines = f_obj.readlines()
	f_obj.close()

	data_lines = lines[1:]
	data_lines = [line.split('\n')[0] for line in data_lines]
	all_data = [line.split(',') for line in data_lines]
	
	X = np.array([line[1:-2] for line in all_data], dtype=float)
	y = np.array(
		[ line[year_index[y_var]] 
		  for line in all_data],
		dtype=int)

	return X, y

