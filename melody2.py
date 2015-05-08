import timeit
import numpy as np
from sklearn import linear_model

def find_percentiles(values, num_splits):

	percentile = [ (i*100.) / num_splits for i in range(1,num_splits)]

	return [np.percentile(values, p) for p in percentile]

class Melody:

	def __init__(self, n_splits=4):

		self.n_splits = n_splits

		#Initializing all regression models
		self.models = {
			'Linear Regression' : linear_model.LinearRegression(),
			'Ridge Regression' : linear_model.Ridge(alpha = 1),
			#'Bayesian Ridge' : linear_model.BayesianRidge(),
			#'OMP CV' : linear_model.OrthogonalMatchingPursuit(),
			'Lasso' : linear_model.Lasso(alpha = 100),
			'LARS' : linear_model.Lars(n_nonzero_coefs=91)
			}

		self.second_models = [
			linear_model.LinearRegression() for i in range(self.n_splits)]

	def fit(self, X, y):

		self.preds = []

		full_start = timeit.default_timer()
		for model_key in self.models:

			model = self.models[model_key]

			print "Fitting model: %s" % model_key
			model_start = timeit.default_timer()
			model.fit(X, y)
			model_end = timeit.default_timer()
			print "Fitting completed in %f seconds" % (model_end-model_start)

			print "Forming predictions"
			model_start = timeit.default_timer()
			new_preds = model.predict(X)
			model_end = timeit.default_timer()
			print "Prediction completed in %f seconds" % (model_end-model_start)
			print

			self.preds.append(new_preds)

		self.preds = np.array(self.preds).transpose()

		self.std_values = np.std(self.preds, axis=1)

		self.percentiles = find_percentiles(self.std_values, self.n_splits)

		for i in range(self.n_splits):

			if i == 0:
				data_indices = self.std_values <= self.percentiles[i]
			elif i == (self.n_splits-1):
				data_indices = self.std_values > self.percentiles[i-1]
			else:
				data_indices = np.logical_and(
					self.std_values > self.percentiles[i-1],
					self.std_values <= self.percentiles[i]
					)

			X2 = self.preds[data_indices,:]
			y2 = y[data_indices]

			print "Fitting Second-Tier Model %d" % (i+1)
			model_start = timeit.default_timer()
			self.second_models[i].fit(X2, y2)
			model_end = timeit.default_timer()
			print "Fitting completed in %f seconds" % (model_end-model_start)
			print

		full_end = timeit.default_timer()
		print "FULL fitting completed in %f seconds" % (full_end-full_start)

	def predict(self, X):

		first_preds = []

		full_start = timeit.default_timer()
		for model_key in self.models:

			model = self.models[model_key]

			print "Forming %s predictions" % (model_key)
			model_start = timeit.default_timer()
			new_preds = model.predict(X)
			model_end = timeit.default_timer()
			print "Prediction completed in %f seconds" % (model_end-model_start)
			print

			first_preds.append(new_preds)

		first_preds = np.array(first_preds).transpose()

		std_values = np.std(first_preds, axis=1)

		final_preds = np.zeros((X.shape[0],))

		for i in range(self.n_splits):

			if i == 0:
				data_indices = std_values <= self.percentiles[i]
			elif i == (self.n_splits-1):
				data_indices = std_values > self.percentiles[i-1]
			else:
				data_indices = np.logical_and(
					std_values > self.percentiles[i-1],
					std_values <= self.percentiles[i]
					)

			if sum(data_indices) == 0:
				continue

			X2 = first_preds[data_indices,:]

			print "Second-Tier Model %d Prediction" % (i+1)
			model_start = timeit.default_timer()
			new_preds = self.second_models[i].predict(X2)
			model_end = timeit.default_timer()
			print "Prediction completed in %f seconds" % (model_end-model_start)
			print

			final_preds[data_indices] = new_preds

		full_end = timeit.default_timer()
		print "FULL prediction completed in %f seconds" % (full_end-full_start)
		return final_preds

		
