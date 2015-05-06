#GP.py

import timeit
import numpy as np
from main import import_file

from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcess
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score

model_names = {
	'GP' : GaussianProcess(),
	'SVR' : SVR(epsilon = 5),
	'ET' : ExtraTreesClassifier(n_estimators=100)
}

def train_model(model_name, X = None, y = None, num_data_points=-1):

	if X is None:
		print "Importing Data..."
		X, y = import_file('train.csv')

	if num_data_points > 0:
		print "Selecting data points.."
		X = X[:num_data_points,]
		y = y[:num_data_points,]


	model = model_names[model_name]

	print "Training Model"
	start = timeit.default_timer()
	model.fit(X, y)
	end = timeit.default_timer()
	print "Training completed in %f seconds" % (end-start)

	return model

def train_GP(X = None, y = None, num_data_points=-1):
	return train_model('GP', X, y, num_data_points)

def train_SVR(X = None, y = None, resolution = 1, num_data_points=-1):
	return train_model('SVR', X, y, num_data_points)

def train_ET(X = None, y = None, num_data_points=-1):
	return train_model('ET', X, y, num_data_points)

def predict_model(model, X = None, y = None, num_data_points=-1):

	if X is None:
		print "Importing Data..."
		X, y = import_file('cv.csv')

	if num_data_points > 0:
		print "Selecting data points..."
		X = X[:num_data_points,]
		y = y[:num_data_points,]

	print "Performing prediction"
	start = timeit.default_timer()
	preds = model.predict(X)
	end = timeit.default_timer()
	print "Prediction completed in %f seconds" % (end-start)

	return preds, y

def score_preds(preds, y):

	print "Accuracy: "
	print accuracy_score(y, preds)
	print

	print "MSE: "
	print mean_squared_error(y, preds)
	print

	print "r^2: "
	print r2_score(y, preds)

def save_preds(preds, outname):

	f = open(outname, 'w+')

	for elem in preds:
		f.write(str(elem))
		f.write('\n')

	f.close()