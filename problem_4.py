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

dim = len(data)


P = matrix([[1.0, 0, 0, 0, 0],
           [0, 1.0, 0, 0, 0],
           [0, 0, 1.0, 0, 0],
           [0, 0, 0, 1.0, 0],
           [0, 0, 0, 0, 0]])

q = matrix(np.zeros(5))

h = matrix(-1 * np.ones((dim,1)))

w = np.array([0, 0, 0, 0, 0])
G = matrix(np.eye(5))
new_G = matrix(np.eye(5))

for _ in range(200-1):
    new_G = np.vstack((new_G,G))

labels = matrix(labels)
new_G = matrix(new_G)

new_G = np.multiply(labels, new_G)
new_G = matrix(new_G, (dim, 5), 'd')


sol = solvers.qp(P, q, new_G, h)
print(sol['x'])