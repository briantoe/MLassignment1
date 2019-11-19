import random
from math import sqrt

import numpy as np
from scipy.spatial.distance import euclidean
from scipy.stats import multivariate_normal


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

def e_step(xi, means, covariances, lams, j):
    # xi is a datapoint
    # means is the means for all the gaussians
    # same with covariances
    # lams is all the lambdas for the gaussians
    # j is an index of a particular lambda

    q = (gauss_prob_density(xi, means[j], covariances[j]) * lams[j]) 
    q = q / (sum([gauss_prob_density(xi, means[l], covariances[l]) for l in range(len(lams))])) 
    
    return q



def m_step(qs):
    pass


def gmm(x, k):
    covariance = [np.eye(len(x[0])) for _ in range(k)]
    mean = [random.random() for _ in range(k)]
    lam = np.random.dirichlet(np.ones(k), size=1)[0]

    qs = []
    for xi in x:
        qs.append([e_step(xi, mean, covariance, lam, j) for j in range(len(lam))])

    
def gauss_prob_density(x, mean, covariance):
    return multivariate_normal.pdf(x, mean, covariance)
    


if __name__ == "__main__":
    x, y = import_data('leaf.data')
    x = normalize_data(x) 



    ks = [12, 18, 24, 36, 42]
    # for k in ks:
    #     objectives = [] 
    # for _ in range(20):

    gmm(x, 12)
