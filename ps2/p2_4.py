# coding:utf-8
import math
import numpy as np
from copy import deepcopy

class DecisionTree:
	def __init__(self):
		self.X = []
		self.Y = []
		self.Tree = []

	def fit(self,X,Y,i):
		self.X = X
		self.Y = Y
		self.Tree = self.build_tree(X,Y,i)

	def sub_(self,X,Y):
		types = list(np.unique(Y))
		# pure -> stop
		if len(types) == 1:
			return types[0]
		else:
			type_size = []
			tmpY = Y[0]
			for t in types:
				type_size.append( len(tmpY[tmpY==t]) )
			major_type = types[type_size.index(max(type_size))]
			return major_type

	# aF - available Feature ID
	def build_tree(self,X,Y,i):
		split_idx = i
		tree_node = {}
		tree_node['idx'] = split_idx
		tree_node['branches'] ={}
		sp_types = np.unique(X[split_idx,:])
		for st in sp_types:
			selected = X[split_idx,:] == st
			sX = X[:,selected]
			sY = Y[:,selected]
			tree_node['major_info']= (Y.shape[1],Y[Y=='p'].size)
			tree_node['branches'][st] = self.sub_(sX,sY)
		return tree_node

	def predict(self,X):
		all_types = list(np.unique(self.Y))
		m,n = X.shape
		ret = []
		for i in range(n):
			x_i = X[:,i]
			cur_node = self.Tree
			while (cur_node not in all_types):
				split_idx = cur_node['idx']
				branch = x_i[split_idx]
				cur_node = cur_node['branches'][branch]
			ret.append(cur_node)
		ret = np.array(ret).reshape(1,n)
		return ret

	def dump_tree(self,node = 0,path=""):
		all_types = list(np.unique(self.Y))
		if node == 0:
			node = self.Tree
		if node in all_types:
			print(path,node)
			return
		split_idx = node['idx']
		ori_path = deepcopy(path)
		for k,v in node['branches'].items():
			new_path = path + "%d:%s -> " % (split_idx,k)
			self.dump_tree(v,new_path)
		return


# O - obtained class from algorithm
def get_accuracy(Y,O):
	n_right = np.sum(Y==O)
	accuracy = float(n_right)/Y.size
	return accuracy


def import_data(fname):
	fh = open(fname,'r')
	content = fh.readlines()
	fh.close()
	X=[];Y=[]
	for line in content:
		values = line.strip().split(',')
		X.append(values[1:])
		Y.append(values[0])
	X= np.array(X).T
	Y= np.array([Y])
	return X,Y


if __name__ == '__main__':
	X_tr,Y_tr = import_data('mush_train.data')
	X_ts,Y_ts = import_data('mush_test.data')
	dtree = DecisionTree()
	N_features = X_tr.shape[0]
	for i in range(N_features):
		dtree.fit(X_tr,Y_tr,i)
		O = dtree.predict(X_ts)
		print(i,get_accuracy(O,Y_ts))








