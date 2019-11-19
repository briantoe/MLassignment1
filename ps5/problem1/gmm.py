import numpy as np
from math import *
import random
from scipy.spatial.distance import euclidean


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


def gmm(x, k):
    
    pass

if __name__ == "__main__":
    x, y = import_data('leaf.data')
    x = normalize_data(x)
    

    ks = [12, 18, 24, 36, 42]
    

