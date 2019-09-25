import numpy as np
from cvxopt import solvers, matrix
import math

np.set_printoptions(threshold=1000)
solvers.options['show_progress'] = True


def read_data(filename):
    data = []
    with open(filename) as file:
        for line in file:
            data_line = [float(item) for item in line.split(',')]
            data.append(data_line)
    data = np.array(data)
    labels = data[:, 0]  # grab labels
    dim = len(data[0])  # dimension of the data
    data = data[:, 1:dim]  # grab vectors

    return (data, labels)


def reform_labels(labels):
    for i in range(len(labels)):
        if labels[i] == 0:
            labels[i] = -1

    return labels


# training step
sigmas = [math.pow(10, i) for i in range(-1, 4)]
cvals = [math.pow(10, i) for i in range(9)]
c = cvals[0]

filename = "park_train.data"

raw_data = read_data(filename)
data = raw_data[0]
labels = raw_data[1]
labels = reform_labels(labels)
dim = len(data[0]) + 1  # + 1 because of b

# for c in cvals:
#     for sigma in sigmas:

P = []
for y_out, x_out in zip(labels, data):
    P_row = []
    for y_in, x_in in zip(labels, data):
        new_y = y_out * y_in
        # apply gaussian kernel function to the x's in xTx
        # K(x1,x2) = exp( || x-z || ^ 2 / (2 * sigma^2))
        # this will be the new_x
        new_x = np.dot(np.transpose(x_out), x_in)
        P_row.append(new_x * new_y)

    P.append(P_row)

q = -1 * np.ones(len(data))
b = np.zeros(len(data))
h_zeros = np.zeros(len(data))
h_cs = np.array([c for i in range(len(data))])
h = np.append(h_zeros, h_cs)

A = np.transpose(np.zeros(len(data)))
b = np.zeros(1)

G_top = -1 * np.eye(len(data))
G_bot = np.eye(len(data))
G = np.vstack((G_top, G_bot))

G = matrix(G)
P = matrix(P)
q = matrix(q)
h = matrix(h)
A = matrix(A)
b = matrix(b)


sol = solvers.qp(P, q, G, h)
sol_arr = np.array(sol['x'])
lams = sol['x']

print(lams)

sum = 0
for lam, y, x in zip(lams, labels, data):
    sum += -1 * lam * y * x

w = -1 * sum
print(w)
 