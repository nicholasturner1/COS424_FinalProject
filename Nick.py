#GP.py

import timeit
import numpy as np
from main import import_file

from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcess
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.manifold import Isomap
from sklearn.lda import LDA

import melody2 as m

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

def train_KNN(X = None, y = None, num_data_points=-1, n_neighbors=1):
	return train_model(
		KNeighborsClassifier(n_neighbors=n_neighbors),
		X, y, num_data_points)

def train_Melody(X = None, y = None, num_data_points=-1, n_splits=4):
	return train_model(
		m.Melody(n_splits),
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

#SURPRISE SURPRISE THIS TAKES TOO LONG
def isomap_data(X, n_components=2, n_neighbors=5, max_iter=None, num_data_points=-1):

	mapping = Isomap(n_neighbors=n_neighbors, n_components=n_components, 
		max_iter=max_iter)

	if num_data_points > 0:
		X = X[:num_data_points,:]

	print "Performing mapping"
	start = timeit.default_timer()
	mapped = mapping.fit_transform(X)	
	end = timeit.default_timer()
	print "Mapping completed in %f seconds" % (end-start)

	return mapped, mapping

def lda_data(X, y, n_components=2, num_data_points=-1):

	lda = LDA(n_components=n_components)

	if num_data_points > 0:
		X = X[:num_data_points,:]
		y = y[:num_data_points]

	print "Performing mapping"
	start = timeit.default_timer()
	mapped = lda.fit_transform(X, y)	
	end = timeit.default_timer()
	print "Mapping completed in %f seconds" % (end-start)

	return mapped, lda



def save_data(X, outname):

	f = open(outname, 'w+')

	for row in X:
		next_line = ','.join(map(str,row))
		f.write(next_line)
		f.write('\n')

	f.close()