# coding:utf-8
import math
import numpy as np
from cvxopt import solvers,matrix
import copy
solvers.options['show_progress'] = False

def reform_labels(labels):
	for i in range(len(labels)):
		if labels[i] == 2:
			labels[i] = -1
		# elif labels[i] == 1:
			# labels[i] = -1

	return labels

def import_data(fname):
	all_data = np.loadtxt(fname=fname, delimiter=',')
	labels = all_data[:,-1]
	data = all_data[: , 0:-1]
	return data, reform_labels(labels)

# c - the degree of slackness
def slack_svm(X,Y,c):
	n_feature = len(X[:,0])
	n_sample = Y.size
	n_paras = n_feature + 1 + n_sample
	# construct P
	P = np.zeros(n_paras)
	for i in range(n_feature):
		P[i]=1
	P = np.diag(P)

	# construct q
	q = np.zeros(n_paras)
	for i in range(n_sample):
		q[n_feature+1+i]=c

	# construct G phase 1, consider y(wx+b)>=1-ksi
	G = []

	for i in range(n_sample):
		#form: y_i*x_i,y_i,0..0,1,0..0
		tmp = np.zeros(n_paras)
		x_i = X[:,i]
		y_i = Y[i]
		tmp[0:n_feature] = y_i*x_i
		tmp[n_feature] = y_i
		tmp[n_feature+1+i] = 1
		G.append(tmp)

	# construct G phase 2, consider ksi >= 0
	for i in range(n_sample):
		tmp = np.zeros(n_paras)
		tmp[n_feature+1+i] =1
		G.append(tmp)
	G = np.array(G)

	# construct h
	h=np.zeros(n_sample*2)
	for i in range(n_sample):
		h[i]=1

	# transform Gx >= h to Gx <= h
	G=-G;h=-h
	ret = solvers.qp(matrix(P),matrix(q),matrix(G),matrix(h))
	solution = ret['x']

	# decompose solution to w,b,ksi
	w = solution[0:n_feature]
	w = np.array(w).reshape(n_feature,1)
	b = solution[n_feature]
	ksi = list(solution[n_feature+1:])
	# verify(X,Y,w,b,ksi)
	return w,b,ksi

# f(x) = wx+b
def F(w,b,x):
	return np.dot(w.T,x)+b

# O - obtained class from algorithm
def get_accuracy(Y,O):
	R = np.multiply(Y.T,O)
	n_right = len(R[R>0])
	accuracy = float(n_right)/len(Y)
	return accuracy

# check the correctness of result parameter
def verify(X,Y,w,b,ksi):
	n_sample = len(Y)
	for i in range(n_sample):
		y_i = Y[i]
		x_i = X[:,i]
		if y_i*(np.dot(w.T,x_i)+b) + ksi[i] < 1:
			print("ERROR FIND !")
			exit(0)
	print("Result PASS!")
	return 0



def pca(data):
    M = np.mean(data.T, axis=1)
    C = data - M
    WWT = np.cov(C.T)
    
    eig_vals, eig_vect = np.linalg.eig(WWT)
    # print(eig_vals, eig_vect)
    # print(sorted(eig_vals, reverse=True))

    return (sorted(eig_vals, reverse=True), [x for _,x in sorted(zip(eig_vals,eig_vect), reverse=True)])
   

if __name__ == '__main__':

	C_list = [1, 10, 100, 1000]
	k_list = [1,2,3,4,5,6]
	accuracy = {'t':[],'v':[]}
	X_t,Y_t = import_data('../sonar_train.data')
	X_v,Y_v = import_data('../sonar_valid.data')
	X_test,Y_test = import_data('../sonar_test.data')
	eig_vals_train, eig_vects_train = pca(X_t)
	# eig_vals_valid, eig_vects_valid = pca(X_v)
	# eig_vals_test, eig_vects_test = pca(X_test)
	bestk = None

	temp = np.array(eig_vects_train)
	PM = None

	for k in k_list:
		X_t,Y_t = import_data('../sonar_train.data')
		X_v,Y_v = import_data('../sonar_valid.data')
		PM = temp.T[0:k]
		X_t = np.dot(PM, X_t.T)
		X_v = np.dot(PM, X_v.T)
 
		for c in C_list:
			w,b,ksi = slack_svm(X_t,Y_t,c)
		
			# print("X train")
			O = F(w,b,X_t)
			accuracy['t'].append(get_accuracy(Y_t,O))
			# print("X valid")
			O = F(w,b,X_v)
			accuracy['v'].append(get_accuracy(Y_v,O))
	
		tmp = accuracy['v']
		# find best parameter combination
		max_accuracy = max(tmp)
		max_configs = list(filter(lambda x:x[1]==max_accuracy, zip(C_list,tmp) ))
	

		# print("\n1.1(b) accuracy",accuracy['t'])
		# print("\n1.1(c) accuracy",accuracy['v'])

		print("\nK = " + str(k))
		print("Accuracy on training data for C = " + str(C_list) + " \n" + str(accuracy['t']))
		print("Accuracy on validation data for C = " + str(C_list) + " \n" + str(accuracy['v']))
		print()
		accuracy['t'] = []
		accuracy['v'] = []


	quit()
	# verify best para on test set
	X_test = np.dot(PM, X_test.T)
	for c,acc in max_configs:
		w,b,ksi = slack_svm(X_t,Y_t,c)
		# print("x test")
	
		O = F(w,b,X_test)
		print("\n1.1(c) Best C ",c)
		print("\t1.1(d) accuracy",get_accuracy(Y_test,O))
