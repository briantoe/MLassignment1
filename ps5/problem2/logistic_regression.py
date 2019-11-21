import numpy as np
from math import log, exp

def import_data(filename):
    all_data = np.loadtxt(fname=filename, delimiter=',')
    labels = all_data[:,0]
    data = all_data[: , 1:]

    return data, labels


def p(yi, xi, w, b):
    return ((yi + 1) / 2) * (np.dot(w, x) + b) - log(1 + exp(np.dot(w, x) + b))


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def update_b(yi, xi, w, b):
    return b + ((yi + 1) / 2) - p(1, xi, w, b)


def update_w(yi, xij, xi, w, b):
    return w + xij * ((yi + 1) / 2) - p(1, xi, w, b)
    



if __name__ == "__main__":
    x, y = import_data('park_train.data')
    # x = normalize_data(x) 
    