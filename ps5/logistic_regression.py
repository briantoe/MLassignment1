import numpy as np
from math import log, exp, pi, sqrt

def normalize_data(x):
    M = np.mean(x.T, axis=1)
    C = x - M

    var = np.var(x.T, axis=1)
   
    x = C
    for i in range(len(x)):
        x[i] = [xi/sqrt(vari) for xi, vari in zip(x[i], var)]

    return x


def import_data(filename):
    all_data = np.loadtxt(fname=filename, delimiter=',')
    labels = all_data[:,0]
    data = all_data[: , 1:]

    return data, labels


def reform_labels(labels):
    for i in range(len(labels)):
        if labels[i] == 0:
            labels[i] = -1

    return labels
    

def log_map_objective(x, y, w, b, lam):
    tot = 0
    for i in range(len(x)):
        if y[i] == 1:
            tot += log(p(x[i], w, b))
        else:
            tot += log(p2(x[i],w,b))

    return tot - (lam / 2.0) * np.linalg.norm(w) ** 2


def p(x, w, b):
    return np.exp(np.dot(w, x)+ b) / (1 + np.exp(np.dot(w, x) + b))

def p2(x,w, b):
    return 1 / (1 + np.exp(np.dot(w, x) + b))


def update_b(x, y, w, b):
    d = [((y[i] + 1 )/2.0) - p(x[i], w, b) for  i in range(len(x))]
    return sum(d)


def update_w(x, y, w, b):
    d = [((y[i] + 1 )/2.0) - p(x[i], w, b) for  i in range(len(x))]
    return np.array([np.dot(x[:, i], d) for i in range(len(x[0]))])


def updates(x, y, w, b):
    new_w = update_w(x, y, w, b)
    new_b = update_b(x, y, w, b)
    return new_w, new_b



def accuracy(data, labels, w, b):
    count = len(data)
    for i in range(len(data)):
        if(-labels[i] * (np.dot(w, data[i]) + b) >= 0):
            count -= 1
    return count


def train(x, y, w, b, maxiters):
    for _ in range(maxiters):
        dw, db = updates(x, y,w,b)
        w += dw
        b += db

    return w, b




if __name__ == "__main__":
    x, y = import_data('park_train.data')
    y = reform_labels(y)
    x = normalize_data(x)
    w = np.zeros(len(x[0]))
    b = 0

    w , b = train(x,y, w,b,3)
    print("Learned w: " , w)
    print("Learned b: ", b)
    print("Accuracy on the training dataset: ", accuracy(x, y, w, b)/ len(x))

    x, y = import_data('park_test.data')
    y = reform_labels(y)
    x = normalize_data(x)
    print("Accuracy on the test dataset: ", accuracy(x, y, w, b)/ len(x))


    x, y = import_data('park_validation.data')
    y = reform_labels(y)
    x = normalize_data(x)
    lams = [0]#[i for i in range(2)]

    objs = [log_map_objective(x, y, w, b, lam) for lam in lams]
    print(max(objs))

