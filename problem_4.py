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

def feature_mapping(x):
    new_x = np.array([1.0])
    new_x = np.append(new_x, list((float(item) for item in x)))
    new_x = np.append(new_x, list((math.pow(item, 2.0) for item in x)))

    return new_x

raw_data = read_data("mystery.data")
data = raw_data[0]
labels = raw_data[1]

w = np.array([0.0, 0.0, 0.0, 0.0])
b = 0.0

n = 5
P = matrix(0.0, (n,n))
P[::n+1] = 1.0
P[-1,-1] = 0.0

q = matrix(0.0, (5,1))
h = matrix([1.0])

for x, y in zip(data, labels):
    G = -y * np.dot(w,x) + b
    
    G = matrix.trans(matrix([G, 0, 0, 0, 0]))
    print(G)
    sol = solvers.qp(P, q, G, h, A=None, b=None)
    print(sol)
