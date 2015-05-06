#GP.py

import timeit
import numpy as np
from main import import_file

from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcess
from sklearn.ensemble import ExtraTreesClassifier

def train_model(model, X = None, y = None, num_data_points=-1):

	if X is None:
		print "Importing Data..."
		X, y = import_file('train_post60.csv')

	if num_data_points > 0:
		print "Selecting data points.."
		X = X[:num_data_points,]
		y = y[:num_data_points,]

	print "Training Model"
	start = timeit.default_timer()
	model.fit(X, y)
	end = timeit.default_timer()
	print "Training completed in %f seconds" % (end-start)

	return model

def train_GP(X = None, y = None, num_data_points=-1):
	return train_model(
		GaussianProcess(), 
		X, y, num_data_points)

def train_SVR(X = None, y = None, resolution = 1, num_data_points=-1):
	return train_model(
		SVR(epsilon=5), 
		X, y, num_data_points)

def train_ET(X = None, y = None, num_data_points=-1, n_estimators=10):
	return train_model(
		ExtraTreesClassifier(n_estimators=n_estimators), 
		X, y, num_data_points)

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
