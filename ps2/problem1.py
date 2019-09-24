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


P = np.zeros((dim + len(data), dim + len(data)))
i = 0
while i < 22:  # creates P matrix of 1s and then 0s on diagonal
    P[i,i] = 1.0    
    i += 1 
P = 2 * P # 2P because of the 1/2 in std form
P = matrix(P)

q = np.zeros(dim)
q = (np.append(q, np.ones(len(data))))
q = matrix(q)

G = np.zeros((len(data), dim))

h_neg_ones = -1 * np.ones(len(data))
h_zeros = np.zeros(len(data))
h = np.append(h_neg_ones, h_zeros)
h = matrix(h)

for row in range(len(data)):
    G[row] = np.hstack((-1 * labels[row] * data[row], -1 * labels[row]))

neg_I = -1 * np.eye(len(data))
G_zeros = np.zeros((len(data), dim))

G_bot = np.hstack((G_zeros, neg_I))
G_top = np.hstack((G, neg_I))

G_final = np.vstack((G_top, G_bot))
G_final = matrix(G_final)


# validation step
filename = "park_validation.data"

raw_data = read_data(filename)
data = raw_data[0]
labels = raw_data[1]
labels = reform_labels(labels)

dim = len(data[0]) + 1 # + 1 because of b

cvals = [math.pow(10, i) for i in range (9)]
bestc= -1
leastmisclassifications = float('inf')
for c in cvals:
    sol = solvers.qp(P, c * q, G_final, h)
    sol_arr = np.array(sol['x'])
    w = sol_arr[:dim-1]
    b = sol_arr[dim-1]
    zi = sol_arr[dim+1-1:]

    misclassified = 0
    for point, label in zip(data, labels):

        if label * ((np.dot(np.transpose(w),point)) + b) <= 0:
            misclassified += 1

    if misclassified < leastmisclassifications:
        leastmisclassifications = misclassified
        bestc = c
    # percent = float((len(data) - misclassified) / len(data)) * 100.0    
    # print("Value of C = %d, amount of misclassifications = %d, Accuracy: %.2f%%" % (c, misclassified, percent))
    # print("Classifier for C = %d" % c)
    # print("w = " + str(w) + '\n')
    # print("b = " + str(b) + '\n')
    # print("zi = " + str(zi)+ '\n')


sol = solvers.qp(P, bestc * q, G_final, h)
sol_arr = np.array(sol['x'])
w = sol_arr[:dim-1]
b = sol_arr[dim-1]
zi = sol_arr[dim+1-1:]

# print("\nBEST C = %d" % (bestc))

# print("\nw: " + str(w) + '\n')
# print("b: " + str(b))
# print("\nzi: " + str(zi) + '\n')

filename = "park_test.data"

raw_data = read_data(filename)
data = raw_data[0]
labels = raw_data[1]
labels = reform_labels(labels)

dim = len(data[0]) + 1 # + 1 because of b

misclassified = 0
for point, label in zip(data, labels):
    if label * ((np.dot(np.transpose(w),point)) + b) <= 0:
        misclassified += 1


percent = float((len(data) - misclassified) / len(data)) * 100.0    
print("Value of C = %d, amount of misclassifications = %d, Accuracy: %.2f%%" % (bestc, misclassified, percent))

sol = solvers.qp(P, bestc * q, G_final, h)
sol_arr = np.array(sol['x'])
w = sol_arr[:dim-1]
b = sol_arr[dim-1]
zi = sol_arr[dim+1-1:]
