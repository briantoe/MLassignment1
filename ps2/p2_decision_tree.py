# coding:utf-8
import math
import numpy as np
from copy import deepcopy

class DecisionTree:
	def __init__(self):
		self.X = []
		self.Y = []
		self.Tree = []

	def fit(self,X,Y):
		self.X = X
		self.Y = Y
		n_features = X.shape[0]
		avaliable_features = [i for i in range(n_features)]
		self.Tree = self.build_tree(X,Y,avaliable_features)

	def get_major_type(self,X,Y):
		types = list(np.unique(Y))
		type_size = []
		tmpY = Y[0]
		for t in types:
			type_size.append( len(tmpY[tmpY==t]) )
		max_val = max(type_size)
		major_type = types[type_size.index(max_val)]

		type_size = np.array(type_size)
		assert (np.sum(type_size == max_val) ==1)

		return major_type

	# aF - available Feature ID
	def build_tree(self,X,Y,aF):
		types = list(np.unique(Y))
		# handle empty leaf
		if X.size == 0:
			return 'vote'
		# pure -> stop
		if len(types) == 1:
			return types[0]
		# not pure but run out of attributes -> majority vote
		elif len(aF) == 0:
			return self.get_major_type(X,Y)

		aF.sort(reverse=True)
		info_gain = []
		for i in aF:
			info_gain.append(self.calculate_info_gain(X,Y,i))

		# since aF is reversed, split_idx is last if we have tie
		split_idx = aF[info_gain.index(max(info_gain))]
		print("Split at feature {}, IG is {}".format(split_idx,max(info_gain)))
		tree_node = {}
		tree_node['idx'] = split_idx
		saF = deepcopy(aF)
		saF.remove(split_idx)
		tree_node['branches'] ={}
		sp_types = np.unique(self.X[split_idx,:])
		for st in sp_types:
			selected = X[split_idx,:] == st
			sX = X[:,selected]
			sY = Y[:,selected]
			sub_tree = self.build_tree(sX,sY,saF)
			if sub_tree == 'vote':
				sub_tree = self.get_major_type(X,Y)
			tree_node['branches'][st] = sub_tree
		return tree_node

	def predict(self,X):
		all_types = list(np.unique(self.Y))
		_,n = X.shape
		ret = []
		for i in range(n):
			x_i = X[:,i]
			# todo: handle the situation that we have not
			# seen all branches in the train set
			cur_node = self.Tree
			while (cur_node not in all_types):
				split_idx = cur_node['idx']
				branch = x_i[split_idx]
				cur_node = cur_node['branches'][branch]
			ret.append(cur_node)
		ret = np.array(ret).reshape(1,n)
		return ret

	# calculate the entropy
	def calculate_entropy(self,Y):
		types = list(np.unique(Y))
		n_samples = Y.shape[1]
		# all sample has some label, entropy is 0
		if len(types) == 1:
			return 0.0

		entropy = 0.0
		tmpY = Y[0]
		for t in types:
			prob = len(tmpY[tmpY==t])/float(n_samples)
			entropy += -prob*math.log(prob)
		return entropy

	# calculate the information gain if we split on i^{th} attr
	def calculate_info_gain(self,X,Y,i):
		ori_entropy = self.calculate_entropy(Y)
		new_entropy = 0
		sp_types = np.unique(X[i,:])
		for st in sp_types:
			selected = X[i,:] == st
			sY = Y[:,selected]
			ratio = sY.size /float(Y.size)
			new_entropy += ratio*self.calculate_entropy(sY)
		info_gain = ori_entropy - new_entropy
		return info_gain

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
			new_path = ori_path + "%d:%s -> " % (split_idx,k)
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
	dctree = DecisionTree()
	dctree.fit(X_tr,Y_tr)
	O = dctree.predict(X_tr)
	tr_accuracy = get_accuracy(Y_tr,O)
	O = dctree.predict(X_ts)
	ts_accuracy = get_accuracy(Y_ts,O)

	print("\n2.1 the Tree is\n")
	dctree.dump_tree()
	print("\n2.4 train accuracy",tr_accuracy)
	print("\n2.5 test accuracy",ts_accuracy)