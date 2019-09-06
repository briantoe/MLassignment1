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
    labels = data[:, -1] # grab all the labels
    data = data[:, 0:4] # grab all the vectors

    return (data, labels)

def feature_mapping(old_data):
    mapped_data = []
    for old_row in old_data:
        new_row = list(old_row[0:4]) + list(math.pow(old_item, 2.0) for old_item in old_row[0:1]) + list(math.pow(old_item, 5.0) for old_item in old_row[1:2]) + list(math.pow(old_item, 3.0) for old_item in old_row[2:4]) + [12]
        mapped_data.append(new_row)

    return (np.array(mapped_data))


raw_data = read_data("mystery.data")
data = raw_data[0]
labels = raw_data[1]

data = feature_mapping(data)
dim = len(data[0]) + 1
print(dim)

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
print(sol['x'])