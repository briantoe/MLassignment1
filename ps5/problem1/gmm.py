import copy
import random
from math import sqrt, log

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
    q = q / (sum([gauss_prob_density(xi, means[l], covariances[l]) * lams[l] for l in range(len(lams))])) 
    return q


def m_step(qs, xs, k):
    lams = np.mean(qs, axis=0)

    means = []
    q_tranpose = np.array(qs).T
    means = np.divide(np.matmul(q_tranpose, np.array(xs)), np.sum(qs, axis=0).reshape(k,1))

    temp = np.sum(qs, axis=0).reshape(k,1)
    covariance = []
    N = len(x)
    for y in range(k):
        num = 0
        den = temp[y]
        for i in range(N):
            vect = np.array(xs[i] - means[y])
            num += qs[i][y] * np.outer(vect, vect.T)

        temp_cov = np.divide(num, np.sum(den))
        covariance.append(temp_cov)

    return means, covariance, lams


def compute_objective(mean, covariance, lam, q, x, k):
    out = 0
    for i in range(len(x)):
        temp_out = 0
        for y in range(k):
            temp_out += q[i][y]* log(gauss_prob_density(x[i], mean[y], covariance[y]) / q[i][y])
        out += temp_out
    return out

def gauss_prob_density(x, mean, covariance):
    return multivariate_normal.pdf(x, mean, covariance)


def gmm(x, k):
    prev_cov = []
    prev_mean = []
    prev_lam = []
    prev_qs = []
    covariance = [np.eye(len(x[0])) for _ in range(k)]
    mean = np.zeros((k, len(x[0])))
    lam = np.random.dirichlet(np.ones(k), size=1)[0]

    while True:
        # e_step
        qs = []
        for xi in x:
            qs.append([e_step(xi, mean, covariance, lam, j) for j in range(len(lam))])

        # m_step
        mean, covariance, lam = m_step(qs,x,k)

        if np.array_equal(prev_cov, covariance) and np.array_equal(prev_mean, mean) and np.array_equal(prev_lam, lam) and np.array_equal(prev_qs, qs):
            break
        
        # keep track of previous iteration
        prev_cov = copy.deepcopy(covariance)
        prev_mean = copy.deepcopy(mean)
        prev_lam = copy.deepcopy(lam)
        prev_qs = copy.deepcopy(qs)

    return mean, covariance, lam, qs
    

if __name__ == "__main__":
    x, y = import_data('leaf.data')
    x = normalize_data(x) 
    
    ks = [12, 18, 24, 36, 42]
    for k in ks:
        objectives = [] 
        for _ in range(20):
            mean, covariance, lam, qs = gmm(x, 12)
            objectives.append(compute_objective(mean, covariance, lam, qs, x, k))

        print("k = %d \t mean = %f variance = %f" % (k, np.mean(objectives), np.var(objectives)))
