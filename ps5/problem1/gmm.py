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

def kmeans(x, y, k):
    means = [] # mus
    clusters = dict() # S
    for i in range(k):
        means.append([random.uniform(-3,3) for _ in range(len(x[0]))])
        clusters[i] = []

    for datapoint in x:
        dists = []
        for mean in means:
            dists.append(euclidean(datapoint, mean))
        m = min(dists)
        clusters[dists.index(m)].append(datapoint)
        
    for key in clusters:
        if len(clusters[key]) == 0: 
            continue
        else:
            means[key] = np.mean(clusters[key])
    
    

if __name__ == "__main__":
    x, y = import_data('leaf.data')
    x = normalize_data(x)
    

    ks = [12, 18, 24, 36, 42]
    kmeans(x, y, 12)
    # for k in ks:
        # kmeans(x, y, k)
