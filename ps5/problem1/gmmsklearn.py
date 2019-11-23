import numpy as np
import sklearn as skl
from sklearn.mixture import GaussianMixture
from math import sqrt
import random


def import_data(filename):
    all_data = np.loadtxt(fname=filename, delimiter=',')
    labels = all_data[:,0]
    data = all_data[: , 1:]

    return data, labels


def normalize_data(x):
    M = np.mean(x.T, axis=1)
    C = x - M

    var = np.var(x.T, axis=1)
   
    x = C
    for i in range(len(x)):
        x[i] = [xi/sqrt(vari) for xi, vari in zip(x[i], var)]

    return x


if __name__ == "__main__":
    x, y = import_data('leaf.data')
    x = normalize_data(x) 
    
    ks = [12, 18, 24, 36, 42]
    for k in ks:
        covariance = [np.eye(len(x[0])) for _ in range(k)]
        mean = [[random.uniform(-3,3) for _ in range(len(x[0]))] for _ in range(k)]

        gmm = GaussianMixture(n_components=k, covariance_type='diag', means_init=mean)
        gmm.fit(x)
        p = gmm.predict(x)
        print(p)