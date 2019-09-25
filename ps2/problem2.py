import numpy as np
from cvxopt import solvers, matrix
import math

np.set_printoptions(threshold=1000)
solvers.options['show_progress'] = False


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

def reform_labels(labels):
    for i in range(len(labels)):
        if labels[i] == 0:
            labels[i] = -1

    return labels


# training step
sigmas = [math.pow(10, i) for i in range(-1,4)]
cvals = [math.pow(10, i) for i in range (9)]


filename = "park_train.data"

raw_data = read_data(filename)
data = raw_data[0]
labels = raw_data[1]
labels = reform_labels(labels)

dim = len(data[0]) + 1 # + 1 because of b

P = np.eye(len(data))
q = -1 * np.ones(len(data))
b = np.zeros(len(data)) 
h_zeros = np.zeros(len(data))
h_cs = np.array([cvals[0] for i in range(len(data))])

G = []

for y, x in zip(labels,data):
    G_row = [x_item * y for x_item in x]
    G.append(G_row)
    
G = np.array(G)
print(G.shape)
G = matrix(G)
P = matrix(P)
q = matrix(q)
h = matrix(h)

sol = solvers.qp(P, q, G, h)
sol_arr = np.array(sol['x'])
w = sol_arr[:dim-1]
b = sol_arr[dim-1]
zi = sol_arr[dim+1-1:]

