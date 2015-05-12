import sys
import os
import math
import bz2
import subprocess
import operator
from decimal import *
from subprocess import Popen, PIPE
from os.path import expanduser
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import numpy as np
from sklearn import datasets, linear_model
import main as m


def main():
    X_train, y_train = m.import_file('../train.csv', "yr5")
    X_cv, y_cv = m.import_file('../test.csv', "yr5")
    # Y axis will have the years, X axis will have the 90 features
    
    ############### With new features
    train_len = len(y_train)
    vector_len = len(X_train[0])
    #print 'train_len, vector_len'
    #print train_len
    #print vector_len
    
    extra_features = 2
    train_added = np.zeros((train_len,vector_len+extra_features), dtype=float)

    for i in range(0,train_len):
	train_added[i] = np.concatenate([X_train[i],[X_train[i,13]/X_train[i,1]],[X_train[i,14]/X_train[i,1]]])
        
    X_train = train_added

    
    test_len = len(y_cv)
    test_vector_len = len(X_cv[0])
    test_added = np.zeros((test_len, test_vector_len + extra_features), dtype=float)
    
    for i in range(0, test_len):
	test_added[i] = np.concatenate([X_cv[i],[X_cv[i,13]/X_cv[i,1]],[X_cv[i,14]/X_cv[i,1]]])
        
    #print X_cv
    X_cv = test_added
    #print X_cv


    # Should automatically do multivariate linear regression
    # Code citation: http://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html
    # Create linear regression object
    regr = linear_model.LinearRegression()
    
    # Train the model using the training sets
    regr.fit(X_train, y_train)
    
    # The coefficients
    print('Coefficients: \n', regr.coef_)

    # do the prediction
    predic = regr.predict(X_cv)

    # The mean square error
    print("Residual sum of squares: %.2f"
          % np.mean((predic - y_cv) ** 2))
    # Explained variance score: 1 is perfect prediction
    print('Explained variance score: 1 is perfect prediction')
    print('Variance score: %.2f' % regr.score(X_cv, y_cv))
    
    # Plot outputs
    #plt.scatter(X_test, y_test,  color='black')
    #plt.plot(X_test, predic, color='blue',
    #         linewidth=3)
    
    #plt.xticks(())
    #plt.yticks(())

    #plt.show()

    # ((predition - actual) data points)^2 element-wise
    # squared error = y, x = years
    subbed = np.subtract(predic, y_cv)
    squaredNow = np.square(subbed)
    print(len(squaredNow))
    print('y_test_post_60 length = ')
    print squaredNow
    print y_cv
    print(len(y_cv))
    
    # Plot outputs
    #plt.scatter(y_cv, squaredNow,  color='black', alpha=.1)
    #plt.plot(X_test_post_60, predic, color='blue',
    #         linewidth=3)
    
    #plt.xticks(())
    #plt.yticks(())
    
    #plt.show()

    m.save_preds(predic, './out/LR_extra2_full_5_test_preds.csv')
    #m.score_preds(predic, y_cv)



    print('done')



if __name__ == '__main__': 
    main() 
