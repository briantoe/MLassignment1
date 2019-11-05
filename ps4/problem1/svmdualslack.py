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
    labels = data[:, 0]  # grab labels
    dim = len(data[0])  # dimension of the data
    data = data[:, 1:dim]  # grab vectors

    return (data, labels)


def reform_labels(labels):
    for i in range(len(labels)):
        if labels[i] == 2:
            labels[i] = -1

    return labels

def validate(w, b):
    filename = "sonar_valid.data"

    raw_data = read_data(filename)
    data = raw_data[0]
    labels = raw_data[1]
    labels = reform_labels(labels)

    leastmisclassifications = float('inf')
    misclassified = 0
    for point, label in zip(data, labels):
        if label * ((np.dot(np.transpose(w),point)) + b) <= 0:
            misclassified += 1

    if misclassified < leastmisclassifications:
        leastmisclassifications = misclassified    

    return (float((len(data) - leastmisclassifications) / len(data)) * 100.0, leastmisclassifications) 

# training step


# filename = "park_train.data"

# raw_data = read_data(filename)
# data = raw_data[0]
# labels = raw_data[1]
# labels = reform_labels(labels)
# dim = len(data[0]) + 1  # + 1 because of b
class SVM:
    def __init__(self):
        self.bestw = None
        self.bestb = None
        pass

    def train(self, d, l):
        data = d
        labels = reform_labels(l)
        dim = len(data[0]) + 1  # + 1 because of b

        sigmas = [math.pow(10, i) for i in range(-1, 4)]
        cvals = [math.pow(10, i) for i in range(9)]
        bestc = -1
        bestsigma = -1
        bestw = None
        bestb = None
        bestpercent = -1.0
        leastmisclassifications = None

        for c in cvals:
            for sigma in sigmas:
                P = []
                for y_out, x_out in zip(labels, data):
                    P_row = []
                    for y_in, x_in in zip(labels, data):
                        new_y = y_out * y_in
                        # apply gaussian kernel function to the x's in xTx
                        # K(x1,x2) = exp( || x-z || ^ 2 / (2 * sigma^2))
                        # this will be the new_x
                        new_x = math.exp(-1 * (np.linalg.norm(np.subtract(x_out, x_in)) ** 2) / (2 * (sigma ** 2)))
                        P_row.append(new_x * new_y)

                    P.append(P_row)

                q = -1 * np.ones(len(data))
                h_zeros = np.zeros(len(data))
                h_cs = np.array([c for i in range(len(data))])
                h = np.append(h_zeros, h_cs)

                A = labels
                A = np.array(A)
                A = np.reshape(A, (len(labels), 1))
                A = np.transpose(A)


                b = matrix(0.0)

                G_top = -1 * np.eye(len(data))
                G_bot = np.eye(len(data))
                G = np.vstack((G_top, G_bot))

                G = matrix(G)
                P = matrix(P)
                q = matrix(q)
                h = matrix(h)
                A = matrix(A)
                b = matrix(b)


                sol = solvers.qp(P, q, G, h, A, b)
                sol_arr = np.array(sol['x'])
                lams = sol['x']

                # print("lams: ")
                # print(lams)
                

                sum = 0
                for lam, temp, x in zip(lams, labels, data):
                    sum += -1 * lam * temp * x

                w = -1 * sum
                # print("w: ")
                # print(w)
            
                cherrypick = 0
                for lam in lams:
                    if lam != c and lam != 0:
                        break 
                    cherrypick += 1
                
                
                b = (1/labels[cherrypick]) - (np.dot(np.transpose(w), data[cherrypick]))
                # print("b: ")
                # print(b)

        def valid(w, b):
            # validation step
            temp = validate(w,b)
            percent = temp[0]
            misclass = temp[1]
            print("C = %d, sigma = %d, misclassifications = %d, accuracy = %.2f%%" % (c, sigma, misclass, percent))
            
            if percent > bestpercent:
                bestc = c
                bestsigma = sigma
                bestpercent = percent
                bestw = w
                bestb = b
                leastmisclassifications = misclass
            # sigma and c correspond with the accuracy
            # for c in cs
            #   for sigma in sigmas


print("Validation: ")
print("Value of BEST C = %d, value of BEST sigma = %d, amount of misclassifications = %d, Accuracy = %.2f%%" % (bestc, bestsigma, leastmisclassifications, bestpercent))


filename = "park_test.data"

raw_data = read_data(filename)
data = raw_data[0]
labels = raw_data[1]
labels = reform_labels(labels)

dim = len(data[0]) + 1 # + 1 because of b

misclassified = 0
for point, label in zip(data, labels):
    if label * ((np.dot(np.transpose(bestw),point)) + bestb) <= 0:
        misclassified += 1


percent = float((len(data) - misclassified) / len(data)) * 100.0  
print("Testing:")
print("Value of C = %d, Value of sigma = %d, amount of misclassifications = %d, Accuracy: %.2f%%" % (bestc, bestsigma, misclassified, percent))
