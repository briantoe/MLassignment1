import numpy as np
from cvxopt import solvers


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


raw_data = read_data("mystery.data")
data = raw_data[0]
labels = raw_data[1]

