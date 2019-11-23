import numpy as np
from math import log, exp, pi

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


def p(yi, xi, w, b):
    return ((yi + 1) / 2.0) * (np.dot(w, xi) + b) - log(1 + exp(np.dot(w, xi) + b))


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def predict(feats, w):
    z = np.dot(feats, w)
    return sigmoid(z)


def cost_function(hxi, yi):
    if yi == 1:
        return -log(hxi)
    elif yi == 0 or yi == -1:
        return -log(1-hxi)


def derivative_sigmoid(z):
    return sigmoid(z) * (1 - sigmoid(z))


def update_b(y, x, w, b):
    return sum([((yi + 1)/2.0) - p(1, xi, w, b) for xi, yi in zip(x, y)])


def update_w(y, x, w, b):
    new_w = []
    for i in range(len(x)):
        wj = 0
        for j in range(len(x[i])):
            wj += x[i][j] * (((y[i] + 1) / 2.0) - p(1, x[i], w, b))
        new_w.append(wj)

    return new_w


def train(x, y, w, b, maxiters):
    
    for _ in range(maxiters):
        for xi, yi in zip(x, y):
            pred = predict(xi, w)
            if(pred >= 0.5 and y == 1) or (pred < 0.5 and y == -1):
                continue
            else:
                b += update_b(y, x, w, b)
                w += update_w(y, x, w, b)

    return w, b



def compute_logMAP_objective(y, x, w, b):
    obj = 0
    for i in range(x):
        obj += log(p(y[i], x[i], w, b))
    
    return obj - 

if __name__ == "__main__":
    x, y = import_data('park_train.data')
    y = reform_labels(y)
    w = [np.random.uniform(-3, 3) for _ in range(len(x[0]))]
    b = np.random.uniform(-3, 3)
