import numpy as np
from math import sqrt
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score

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
		dtype=float)

	return X, y

def score_preds(preds, y, accuracy=False):

	if accuracy:
		print "Accuracy: "
		print accuracy_score(y, preds)
		print

	print "RMSE: "
	print sqrt(mean_squared_error(y, preds))
	print

	print "Mean Absolute Error"
	print np.mean(np.absolute(preds - np.array(y)))
	print

	print "r^2: "
	print r2_score(y, preds)

def save_preds(preds, outname):

	f = open(outname, 'w+')

	for elem in preds:
		f.write(str(elem))
		f.write('\n')

	f.close()