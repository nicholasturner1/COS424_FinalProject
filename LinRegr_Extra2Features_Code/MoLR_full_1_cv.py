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


# 92 features: 
#	13th/1st naturally, start counting at 14th/1st, dot division
#   	horizontally stack; hstackExploring the D



def main():
    X_train, y_train = m.import_file('../train.csv', "yr")
    X_cv, y_cv = m.import_file('../cv.csv', "yr")
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


    #print train_added
    #print len(train_added[1000])
    #print len(train_added)

    #print X_train
    
    
    # Add the two additional features here to training data
    #print('X_train is')
    #print( X_train)

    #elem = X_train[0]
    #print 'elem is'
    #print elem
    #print len(elem)

    #0th feature
    #feat0 = elem[0]
    #print feat0

    #12th feat
    #feat12 = elem[12]
    #print feat12

    #13th feat
    #feat13 = elem[13]
    #print feat13

    # append two additional elements to array elem
    #newOne = np.append(elem, [feat12 / feat0])
    #newOne = np.append(newOne, [feat13 / feat0])
    
    #extra_X_train = np.append(extra_X_train, newOne)
    #print extra_X_train
    
    #print len(elem)
    #print len(newOne)
    #print len(X_train[1])

    #for i in range(0,len(X_train)):   #enumerate(X_train): # in np.nditer(X_train):
        #print i
        #elem = X_train[i]
        
        # 0th feature
        #feat0 = elem[0]
        
        # 12th feat
        #feat12 = elem[12]

        # 13th feat
        #feat13 = elem[13]
        
        # append two additional elements to array elem
        #newOne = np.append(elem, [feat12 / feat0])
        #newOne = np.append(newOne, [feat13 / feat0])
        
        #extra_X_train = np.append(extra_X_train, newOne)
        
    #print len(X_train)
    #print len(extra_X_train)
    #print len(extra_X_train[100])

    
    
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

    m.save_preds(predic, './out/LR_extra2_full_1_cv_preds.csv')
    m.score_preds(predic, y_cv)



    print('done')



if __name__ == '__main__': 
    main() 
