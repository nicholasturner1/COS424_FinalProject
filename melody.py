# COS 424 Final Project


#### Imports
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import preprocessing
from sklearn import svm
from ggplot import *
import pickle

#### Definitions
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

#### Reading the training dataset
X = []
y = []

X, y = import_file("train_post60.csv")
X_t, y_t = import_file("test_post60.csv")

train_len = len(y)
vector_len = len(X[0])

#### Sort by year
train = np.zeros((train_len,vector_len+1), dtype=float)

for i in range(0,train_len):
	train[i] = np.concatenate([[y[i]],X[i]])

sorted_train = train[np.argsort(train[:,0])]

# #### Basic Visual Analysis

# f, ((ax1,ax2,ax3,ax4,ax5), (ax6,ax7,ax8,ax9,ax10), (ax11,ax12,ax13,ax14,ax15)) = plt.subplots(3, 5, sharex='col', sharey='row')
# ax1.scatter(sorted_train[:,0], np.dot(5,sorted_train[:,1]), color='red', label="1")
# ax2.scatter(sorted_train[:,0], sorted_train[:,2], color='red', label="2")
# ax3.scatter(sorted_train[:,0], sorted_train[:,3], color='red', label="3")
# ax4.scatter(sorted_train[:,0], sorted_train[:,4], color='red', label="4")
# ax5.scatter(sorted_train[:,0], sorted_train[:,5], color='red', label="5")
# ax6.scatter(sorted_train[:,0], sorted_train[:,6], color='red', label="6")
# ax7.scatter(sorted_train[:,0], sorted_train[:,7], color='red', label="7")
# ax8.scatter(sorted_train[:,0], sorted_train[:,8], color='red', label="8")
# ax9.scatter(sorted_train[:,0], sorted_train[:,9], color='red', label="9")
# ax10.scatter(sorted_train[:,0], np.dot(2,sorted_train[:,10]), color='red', label="10")
# ax11.scatter(sorted_train[:,0], np.dot(2,sorted_train[:,11]), color='red', label="11")
# ax12.scatter(sorted_train[:,0], np.dot(2,sorted_train[:,12]), color='red', label="12")
# ax13.scatter(sorted_train[:,0], np.dot(0.4,sorted_train[:,13]), color='red', label="13")
# ax14.scatter(sorted_train[:,0], np.dot(0.006,sorted_train[:,14]), color='red', label="14")
# ax15.scatter(sorted_train[:,0], np.dot(0.01,sorted_train[:,15]), color='red', label="15")
# plt.show()


# f, ((ax1,ax2,ax3,ax4,ax5), (ax6,ax7,ax8,ax9,ax10), (ax11,ax12,ax13,ax14,ax15)) = plt.subplots(3, 5, sharex='col', sharey='row')
# ax1.scatter(sorted_train[:,0], np.dot(0.5,sorted_train[:,1+15]), color='red')
# ax2.scatter(sorted_train[:,0], sorted_train[:,2+15], color='red')
# ax3.scatter(sorted_train[:,0], sorted_train[:,3+15], color='red')
# ax4.scatter(sorted_train[:,0], sorted_train[:,4+15], color='red')
# ax5.scatter(sorted_train[:,0], sorted_train[:,5+15], color='red')
# ax6.scatter(sorted_train[:,0], sorted_train[:,6+15], color='red')
# ax7.scatter(sorted_train[:,0], sorted_train[:,7+15], color='red')
# ax8.scatter(sorted_train[:,0], sorted_train[:,8+15], color='red')
# ax9.scatter(sorted_train[:,0], sorted_train[:,9+15], color='red')
# ax10.scatter(sorted_train[:,0], np.dot(1,sorted_train[:,10+15]), color='red')
# ax11.scatter(sorted_train[:,0], np.dot(0.3,sorted_train[:,11+15]), color='red')
# ax12.scatter(sorted_train[:,0], np.dot(0.3,sorted_train[:,12+15]), color='red')
# ax13.scatter(sorted_train[:,0], np.dot(1,sorted_train[:,13+15]), color='red')
# ax14.scatter(sorted_train[:,0], np.dot(1,sorted_train[:,14+15]), color='red')
# ax15.scatter(sorted_train[:,0], np.dot(1,sorted_train[:,15+15]), color='red')
# plt.show()

# f, ((ax1,ax2,ax3,ax4,ax5), (ax6,ax7,ax8,ax9,ax10), (ax11,ax12,ax13,ax14,ax15)) = plt.subplots(3, 5, sharex='col', sharey='row')
# ax1.scatter(sorted_train[:,0], np.dot(1,sorted_train[:,1+30]), color='red')
# ax2.scatter(sorted_train[:,0], sorted_train[:,2+30], color='red')
# ax3.scatter(sorted_train[:,0], sorted_train[:,3+30], color='red')
# ax4.scatter(sorted_train[:,0], sorted_train[:,4+30], color='red')
# ax5.scatter(sorted_train[:,0], sorted_train[:,5+30], color='red')
# ax6.scatter(sorted_train[:,0], sorted_train[:,6+30], color='red')
# ax7.scatter(sorted_train[:,0], sorted_train[:,7+30], color='red')
# ax8.scatter(sorted_train[:,0], sorted_train[:,8+30], color='red')
# ax9.scatter(sorted_train[:,0], sorted_train[:,9+30], color='red')
# ax10.scatter(sorted_train[:,0], np.dot(1,sorted_train[:,10+30]), color='red')
# ax11.scatter(sorted_train[:,0], np.dot(0.3,sorted_train[:,11+30]), color='red')
# ax12.scatter(sorted_train[:,0], np.dot(0.3,sorted_train[:,12+30]), color='red')
# ax13.scatter(sorted_train[:,0], np.dot(0.6,sorted_train[:,13+30]), color='red')
# ax14.scatter(sorted_train[:,0], np.dot(1,sorted_train[:,14+30]), color='red')
# ax15.scatter(sorted_train[:,0], np.dot(1,sorted_train[:,15+30]), color='red')
# plt.show()

# f, ((ax1,ax2,ax3,ax4,ax5), (ax6,ax7,ax8,ax9,ax10), (ax11,ax12,ax13,ax14,ax15)) = plt.subplots(3, 5, sharex='col', sharey='row')
# ax1.scatter(sorted_train[:,0], np.dot(6,sorted_train[:,1+45]), color='red')
# ax2.scatter(sorted_train[:,0], sorted_train[:,2+45], color='red')
# ax3.scatter(sorted_train[:,0], sorted_train[:,3+45], color='red')
# ax4.scatter(sorted_train[:,0], sorted_train[:,4+45], color='red')
# ax5.scatter(sorted_train[:,0], np.dot(5,sorted_train[:,5+45]), color='red')
# ax6.scatter(sorted_train[:,0], sorted_train[:,6+45], color='red')
# ax7.scatter(sorted_train[:,0], sorted_train[:,7+45], color='red')
# ax8.scatter(sorted_train[:,0], sorted_train[:,8+45], color='red')
# ax9.scatter(sorted_train[:,0], sorted_train[:,9+45], color='red')
# ax10.scatter(sorted_train[:,0], np.dot(1,sorted_train[:,10+45]), color='red')
# ax11.scatter(sorted_train[:,0], np.dot(1,sorted_train[:,11+45]), color='red')
# ax12.scatter(sorted_train[:,0], np.dot(1,sorted_train[:,12+45]), color='red')
# ax13.scatter(sorted_train[:,0], np.dot(0.6,sorted_train[:,13+45]), color='red')
# ax14.scatter(sorted_train[:,0], np.dot(1,sorted_train[:,14+45]), color='red')
# ax15.scatter(sorted_train[:,0], np.dot(1,sorted_train[:,15+45]), color='red')
# plt.show()

# f, ((ax1,ax2,ax3,ax4,ax5), (ax6,ax7,ax8,ax9,ax10), (ax11,ax12,ax13,ax14,ax15)) = plt.subplots(3, 5, sharex='col', sharey='row')
# ax1.scatter(sorted_train[:,0], np.dot(1,sorted_train[:,1+60]), color='red')
# ax2.scatter(sorted_train[:,0], sorted_train[:,2+60], color='red')
# ax3.scatter(sorted_train[:,0], sorted_train[:,3+60], color='red')
# ax4.scatter(sorted_train[:,0], np.dot(0.3,sorted_train[:,4+60]), color='red')
# ax5.scatter(sorted_train[:,0], np.dot(0.6,sorted_train[:,5+60]), color='red')
# ax6.scatter(sorted_train[:,0], sorted_train[:,6+60], color='red')
# ax7.scatter(sorted_train[:,0], sorted_train[:,7+60], color='red')
# ax8.scatter(sorted_train[:,0], sorted_train[:,8+60], color='red')
# ax9.scatter(sorted_train[:,0], sorted_train[:,9+60], color='red')
# ax10.scatter(sorted_train[:,0], np.dot(3,sorted_train[:,10+60]), color='red')
# ax11.scatter(sorted_train[:,0], np.dot(1,sorted_train[:,11+60]), color='red')
# ax12.scatter(sorted_train[:,0], np.dot(1,sorted_train[:,12+60]), color='red')
# ax13.scatter(sorted_train[:,0], np.dot(1,sorted_train[:,13+60]), color='red')
# ax14.scatter(sorted_train[:,0], np.dot(1,sorted_train[:,14+60]), color='red')
# ax15.scatter(sorted_train[:,0], np.dot(1,sorted_train[:,15+60]), color='red')
# plt.show()

# f, ((ax1,ax2,ax3,ax4,ax5), (ax6,ax7,ax8,ax9,ax10), (ax11,ax12,ax13,ax14,ax15)) = plt.subplots(3, 5, sharex='col', sharey='row')
# ax1.scatter(sorted_train[:,0], np.dot(5,sorted_train[:,1+75]), color='red')
# ax2.scatter(sorted_train[:,0], sorted_train[:,2+75], color='red')
# ax3.scatter(sorted_train[:,0], sorted_train[:,3+75], color='red')
# ax4.scatter(sorted_train[:,0], np.dot(1,sorted_train[:,4+75]), color='red')
# ax5.scatter(sorted_train[:,0], np.dot(1,sorted_train[:,5+75]), color='red')
# ax6.scatter(sorted_train[:,0], np.dot(2,sorted_train[:,6+75]), color='red')
# ax7.scatter(sorted_train[:,0], sorted_train[:,7+75], color='red')
# ax8.scatter(sorted_train[:,0], sorted_train[:,8+75], color='red')
# ax9.scatter(sorted_train[:,0], sorted_train[:,9+75], color='red')
# ax10.scatter(sorted_train[:,0], np.dot(3,sorted_train[:,10+75]), color='red')
# ax11.scatter(sorted_train[:,0], np.dot(1,sorted_train[:,11+75]), color='red')
# ax12.scatter(sorted_train[:,0], np.dot(1,sorted_train[:,12+75]), color='red')
# ax13.scatter(sorted_train[:,0], np.dot(5,sorted_train[:,13+75]), color='red')
# ax14.scatter(sorted_train[:,0], np.dot(1,sorted_train[:,14+75]), color='red')
# ax15.scatter(sorted_train[:,0], np.dot(4,sorted_train[:,15+75]), color='red')
# plt.show()

# np.dot(5,sorted_train[:,13]/sorted_train[:,1])
# 14/1
# 15/1
# 82-89

# fir = 76
# sec = 83

# f, ((ax1), (ax2), (ax3), (ax4), (ax5), (ax6), (ax7)) = plt.subplots(7, 1, sharex='col', sharey='row')
# ax1.scatter(sorted_train[:,0], np.dot(1,sorted_train[:,fir]+sorted_train[:,sec]), color='blue')
# ax2.scatter(sorted_train[:,0], np.dot(1,sorted_train[:,fir]-sorted_train[:,sec]), color='blue')
# ax3.scatter(sorted_train[:,0], np.dot(1,sorted_train[:,fir]*sorted_train[:,sec]), color='blue')
# ax4.scatter(sorted_train[:,0], np.dot(1,sorted_train[:,fir]/sorted_train[:,sec]), color='blue')
# ax1.scatter(sorted_train[:,0], np.dot(1,sorted_train[:,sec]/sorted_train[:,fir]), color='blue')
# ax6.scatter(sorted_train[:,0], np.dot(1,np.ones(train_len)/sorted_train[:,fir]), color='blue')
# ax7.scatter(sorted_train[:,0], np.dot(1,np.ones(train_len)/sorted_train[:,sec]), color='blue')
# ax1.grid()
# ax2.grid()
# ax3.grid()
# ax4.grid()
# ax5.grid()
# ax6.grid()
# ax7.grid()
# plt.show()


# plt.scatter(sorted_train[:,0], np.dot(1,sorted_train[:,1]+sorted_train[:,2]+sorted_train[:,3]), color='blue')
# plt.grid()
# plt.show()


##### 1, 2, 3, GO!

### Linear Regression

reg_enet = linear_model.ElasticNet(alpha=0.5, l1_ratio=0.7)
reg_ard = linear_model.ARDRegression(compute_score=True)
reg_lr1 = linear_model.LogisticRegression(C=10, penalty='l1', tol=0.01)
reg_lr2 = linear_model.LogisticRegression(C=10, penalty='l2', tol=0.01)

reg_lin = linear_model.LinearRegression()
reg1 = reg_lin
reg1.fit(X,y)
result1 = np.ones((len(y_t),1))
for i in range(0,len(y_t)):
	result1[i] = reg1.predict(X_t[i])

reg_ridge = linear_model.Ridge (alpha = 1)
reg2 = reg_ridge
reg2.fit(X,y)
result2 = np.ones((len(y_t),1))
for i in range(0,len(y_t)):
	result2[i] = reg2.predict(X_t[i])

reg_BR = linear_model.BayesianRidge()
reg3 = reg_BR
reg3.fit(X,y)
result3 = np.ones((len(y_t),1))
for i in range(0,len(y_t)):
	result3[i] = reg3.predict(X_t[i])

reg_lasso = linear_model.Lasso(alpha = 0.01)
reg4 = reg_lasso
reg4.fit(X,y)
result4 = np.ones((len(y_t),1))
for i in range(0,len(y_t)):
	result4[i] = reg4.predict(X_t[i])

reg_omp = linear_model.OrthogonalMatchingPursuitCV()
reg5 = reg_omp
reg5.fit(X,y)
result5 = np.ones((len(y_t),1))
for i in range(0,len(y_t)):
	result5[i] = reg5.predict(X_t[i])

reg_lars = linear_model.Lars(n_nonzero_coefs=85)
reg6 = reg_lars
reg6.fit(X,y)
result6 = np.ones((len(y_t),1))
for i in range(0,len(y_t)):
	result6[i] = reg6.predict(X_t[i])

## Verification
num_of_reg = 6;

test = np.zeros((len(y_t),vector_len+1+num_of_reg), dtype=float)

for i in range(0,len(y_t)):
	test[i] = np.concatenate([[y_t[i]],result1[i],result2[i],result3[i],result4[i],result5[i],result6[i],X_t[i]])

sorted_test = test[np.argsort(test[:,0])]

rs = np.zeros((num_of_reg,1), dtype=float)
comp = np.zeros((num_of_reg,1), dtype=float)
for j in range(0,num_of_reg):
	for i in range(0,len(y_t)):
		rs[j] = rs[j] + (sorted_test[i,0]-sorted_test[i,j+1])**2;
		comp[j] = comp[j] + 1

for j in range(0,num_of_reg):
	print("Results %d" % j)
	print("	Residual Sum of Squares: %.10f" % rs[j])
	mse = rs[j]/comp[j]
	print("	Mean Square Error: %.10f" % mse)

pl.clf()
f, ((ax1,ax2,ax3), (ax4,ax5,ax6) ) = plt.subplots(2, 3, sharex='col', sharey='row')
ax1.plot(sorted_test[:,0], sorted_test[:,0], 'r')
ax1.scatter(sorted_test[:,0], sorted_test[:,1], color='blue')
ax1.grid()
ax2.plot(sorted_test[:,0], sorted_test[:,0], 'r')
ax2.scatter(sorted_test[:,0], sorted_test[:,2], color='blue')
ax2.grid()
ax3.plot(sorted_test[:,0], sorted_test[:,0], 'r')
ax3.scatter(sorted_test[:,0], sorted_test[:,3], color='blue')
ax3.grid()
ax4.plot(sorted_test[:,0], sorted_test[:,0], 'r')
ax4.scatter(sorted_test[:,0], sorted_test[:,4], color='blue')
ax4.grid()
ax5.plot(sorted_test[:,0], sorted_test[:,0], 'r')
ax5.scatter(sorted_test[:,0], sorted_test[:,5], color='blue')
ax5.grid()
ax6.plot(sorted_test[:,0], sorted_test[:,0], 'r')
ax6.scatter(sorted_test[:,0], sorted_test[:,6], color='blue')
ax6.grid()
plt.show()



############### With new features

extra_features = 2
train_added = np.zeros((train_len,vector_len+1+extra_features), dtype=float)

for i in range(0,train_len):
	train_added[i] = np.concatenate([[y[i]],X[i],[X[i,13]/X[i,1]],[X[i,14]/X[i,1]]])

sorted_train_added = train_added[np.argsort(train_added[:,0])]


reg_enet = linear_model.ElasticNet(alpha=0.5, l1_ratio=0.7)
reg_ard = linear_model.ARDRegression(compute_score=True)
reg_lr1 = linear_model.LogisticRegression(C=10, penalty='l1', tol=0.01)
reg_lr2 = linear_model.LogisticRegression(C=10, penalty='l2', tol=0.01)

reg_lin = linear_model.LinearRegression()
reg1 = reg_lin
reg1.fit(train_added[:,1:],train_added[:,0])
resulta1 = np.ones((len(y_t),1))
for i in range(0,len(y_t)):
	resulta1[i] = reg1.predict(np.concatenate([X_t[i],[X_t[i,13]/X_t[i,1]],[X_t[i,14]/X_t[i,1]]]))

reg_ridge = linear_model.Ridge (alpha = 1)
reg2 = reg_ridge
reg2.fit(train_added[:,1:],train_added[:,0])
resulta2 = np.ones((len(y_t),1))
for i in range(0,len(y_t)):
	resulta2[i] = reg2.predict(np.concatenate([X_t[i],[X_t[i,13]/X_t[i,1]],[X_t[i,14]/X_t[i,1]]]))

reg_BR = linear_model.BayesianRidge()
reg3 = reg_BR
reg3.fit(train_added[:,1:],train_added[:,0])
resulta3 = np.ones((len(y_t),1))
for i in range(0,len(y_t)):
	resulta3[i] = reg3.predict(np.concatenate([X_t[i],[X_t[i,13]/X_t[i,1]],[X_t[i,14]/X_t[i,1]]]))

reg_omp = linear_model.OrthogonalMatchingPursuitCV()
reg5 = reg_omp
reg5.fit(train_added[:,1:],train_added[:,0])
resulta5 = np.ones((len(y_t),1))
for i in range(0,len(y_t)):
	resulta5[i] = reg5.predict(np.concatenate([X_t[i],[X_t[i,13]/X_t[i,1]],[X_t[i,14]/X_t[i,1]]]))

reg_lasso = linear_model.Lasso(alpha = 100)
reg6 = reg_lasso
reg6.fit(train_added[:,1:],train_added[:,0])
resulta6 = np.ones((len(y_t),1))
for i in range(0,len(y_t)):
	resulta6[i] = reg6.predict(np.concatenate([X_t[i],[X_t[i,13]/X_t[i,1]],[X_t[i,14]/X_t[i,1]]]))

#85
reg_lars = linear_model.Lars(n_nonzero_coefs=91)
reg4 = reg_lars
reg4.fit(train_added[:,1:],train_added[:,0])
resulta4 = np.ones((len(y_t),1))
for i in range(0,len(y_t)):
	resulta4[i] = reg4.predict(np.concatenate([X_t[i],[X_t[i,13]/X_t[i,1]],[X_t[i,14]/X_t[i,1]]]))

## Verification
num_of_reg = 6;

test = np.zeros((len(y_t),vector_len+1+num_of_reg), dtype=float)

for i in range(0,len(y_t)):
	test[i] = np.concatenate([[y_t[i]],resulta1[i],resulta2[i],resulta3[i],resulta4[i],resulta5[i],resulta6[i],X_t[i]])

sorted_test = test[np.argsort(test[:,0])]

rs = np.zeros((num_of_reg,1), dtype=float)
comp = np.zeros((num_of_reg,1), dtype=float)
for j in range(0,num_of_reg):
	for i in range(0,len(y_t)):
		rs[j] = rs[j] + (sorted_test[i,0]-sorted_test[i,j+1])**2;
		comp[j] = comp[j] + 1

for j in range(0,num_of_reg):
	print("Results %d" % j)
	print("	Residual Sum of Squares: %.10f" % rs[j])
	mse = rs[j]/comp[j]
	print("	RMSE: %.10f" % np.sqrt(mse))


pl.clf()
f, ((ax1,ax2,ax3), (ax4,ax5,ax6) ) = plt.subplots(2, 3, sharex='col', sharey='row')
ax1.plot(sorted_test[:,0], sorted_test[:,0], 'r')
ax1.scatter(sorted_test[:,0], sorted_test[:,1], color='blue')
ax1.grid()
ax2.plot(sorted_test[:,0], sorted_test[:,0], 'r')
ax2.scatter(sorted_test[:,0], sorted_test[:,2], color='blue')
ax2.grid()
ax3.plot(sorted_test[:,0], sorted_test[:,0], 'r')
ax3.scatter(sorted_test[:,0], sorted_test[:,3], color='blue')
ax3.grid()
ax4.plot(sorted_test[:,0], sorted_test[:,0], 'r')
ax4.scatter(sorted_test[:,0], sorted_test[:,4], color='blue')
ax4.grid()
ax5.plot(sorted_test[:,0], sorted_test[:,0], 'r')
ax5.scatter(sorted_test[:,0], sorted_test[:,5], color='blue')
ax5.grid()
ax6.plot(sorted_test[:,0], sorted_test[:,0], 'r')
ax6.scatter(sorted_test[:,0], sorted_test[:,6], color='blue')
ax6.grid()
plt.show()

#### Performing prediction calibration

avg_result = np.ones((len(y_t),1), dtype=float)
res_std = np.ones((len(y_t),1))
flag = np.ones((len(y_t),1))
for i in range(0,len(y_t)):
	# avg_result[i] = np.mean(sorted_test[i,1:j+1])
	res_std[i] = np.std(sorted_test[i,1:num_of_reg+1])
	if res_std[i]>7:
		avg_result[i] = 1.0*1.0*np.mean(sorted_test[i,6])
		#1
		flag[i] = 2020
	elif res_std[i]>5:
		alpha = 0.7
		avg_result[i] = 1.0*2004.0*((1-alpha)*np.mean(sorted_test[i,1:num_of_reg-1])+alpha*np.mean(sorted_test[i,6]))/2005
		flag[i] = 2005
	elif res_std[i]>3:
		alpha = 0.15
		avg_result[i] = 1.0*((1-alpha)*np.mean(sorted_test[i,1:num_of_reg-1])+alpha*np.mean(sorted_test[i,6]))
		flag[i] = 1990
	elif res_std[i]>1:
		avg_result[i] = 1.0*np.mean(sorted_test[i,1:num_of_reg-1])
		flag[i] = 1975
	else:
		avg_result[i] = 1.0*np.mean(sorted_test[i,1:num_of_reg-1])
		flag[i] = 1960

rs = 0
comp = 0

for i in range(0,len(y_t)):
	rs = rs + (sorted_test[i,0]-avg_result[i])**2;
	comp = comp + 1

print("** Residual Sum of Squares: %.10f" % rs)
mse = rs/comp
print("** RMSE: %.10f" % np.sqrt(mse))


plt.plot(sorted_test[:,0], sorted_test[:,0], 'r')
plt.scatter(sorted_test[:,0], avg_result, color='blue')
# plt.scatter(sorted_test[:,0], flag, color='orange')
plt.grid()
plt.show()

# # Storing the results
# with open('avg_result.pickle', 'w') as f:
# 	pickle.dump([av
# 		g_result], f)	


### Special Plot
# shifts = [-100,-50,0,50,100]

# avg_result = np.ones((len(y_t),1), dtype=float)
# res_std = np.ones((len(y_t),1))
# flag = np.ones((len(y_t),1))
# for i in range(0,len(y_t)):
# 	# avg_result[i] = np.mean(sorted_test[i,1:j+1])
# 	res_std[i] = np.std(sorted_test[i,1:num_of_reg+1])
# 	if res_std[i]>7:
# 		avg_result[i] = shifts[4] + 1.0*1.0*np.mean(sorted_test[i,6])
# 		#1
# 		flag[i] = 2020
# 	elif res_std[i]>5:
# 		alpha = 0.7
# 		avg_result[i] = shifts[3] + 1.0*2004.0*((1-alpha)*np.mean(sorted_test[i,1:num_of_reg-1])+alpha*np.mean(sorted_test[i,6]))/2005
# 		flag[i] = 2005
# 	elif res_std[i]>3:
# 		alpha = 0.15
# 		avg_result[i] = shifts[2] + 1.0*((1-alpha)*np.mean(sorted_test[i,1:num_of_reg-1])+alpha*np.mean(sorted_test[i,6]))
# 		flag[i] = 1990
# 	elif res_std[i]>1:
# 		avg_result[i] = shifts[1] + 1.0*np.mean(sorted_test[i,1:num_of_reg-1])
# 		flag[i] = 1975
# 	else:
# 		avg_result[i] = shifts[0] + 1.0*np.mean(sorted_test[i,1:num_of_reg-1])
# 		flag[i] = 1960

# plt.plot(sorted_test[:,0], np.dot(shifts[0],np.ones(len(y_t)))+sorted_test[:,0], 'r')
# plt.plot(sorted_test[:,0], np.dot(shifts[1],np.ones(len(y_t)))+sorted_test[:,0], 'r')
# plt.plot(sorted_test[:,0], np.dot(shifts[2],np.ones(len(y_t)))+sorted_test[:,0], 'r')
# plt.plot(sorted_test[:,0], np.dot(shifts[3],np.ones(len(y_t)))+sorted_test[:,0], 'r')
# plt.plot(sorted_test[:,0], np.dot(shifts[4],np.ones(len(y_t)))+sorted_test[:,0], 'r')
# plt.scatter(sorted_test[:,0], avg_result, color='blue')
# plt.grid()
# plt.show()


# plt.scatter(sorted_test[:,0], res_std, color='blue')
# plt.grid()
# plt.show()


