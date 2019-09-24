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

filename = "park_train.data"

raw_data = read_data(filename)
data = raw_data[0]
labels = raw_data[1]
labels = reform_labels(labels)

dim = len(data[0]) + 1 # + 1 because of b
