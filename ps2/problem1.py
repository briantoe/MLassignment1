import numpy as np
from cvxopt import solvers, matrix
import math


def read_data(filename):
    data = []
    with open(filename) as file:
        for line in file:
            data_line = [float(item) for item in line.split(',')]
            data.append(data_line)
    data = np.array(data)
    labels = data[:, 0] # grab labels
    dim = len(data[0]) # dimension of the data
    data = data[:, 1:dim] # grab vectors

    return (data, labels)


def feature_mapping(old_data):
    mapped_data = []
    for old_row in old_data:
        new_row = list(math.pow(old_item, 3.0) for old_item in old_row[0:4]) +  list((math.pow(old_item, 2.0) * np.exp(old_item)) for old_item in old_row[0:4])#+ list(math.pow(old_item, 2.0) for old_item in old_row[0:4]) #+ list(math.pow(old_item, 5.0) for old_item in old_row[1:2]) + list(math.pow(old_item, 3.0) for old_item in old_row[2:4]) + [12]
        mapped_data.append(new_row)

    return (np.array(mapped_data))

def reform_labels(labels):
    for i in range(len(labels)):
        if labels[i] == 0:
            labels[i] = -1

    return labels

raw_data = read_data("park_train.data")
data = raw_data[0]
labels = raw_data[1]
labels = reform_labels(labels)

dim = len(data[0]) + 1 # + 1 because why??


P = np.eye(dim)
P[-1,-1] = 0.0
P = 2 * P

P = matrix(P)
q = matrix(np.zeros(dim))
h = matrix(-1 * np.ones((len(data),1)))
G = np.zeros((len(data), dim))

for row in range(len(data)):
    G[row] = np.hstack((-1 * labels[row] * data[row], -1 * labels[row]))

G = matrix(G)

sol = solvers.qp(P, q, G, h)

sol_arr = np.array(sol['x'])
w = sol_arr[:dim-1]

b = sol_arr[-1]

print("\nw: " + str(w) + '\n')
print("b: " + str(b))

