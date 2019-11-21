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

def e_step(xi, mean, covariance, lam, j):
    # xi is a datapoint
    # means is the means for all the gaussians
    # same with covariances
    # lams is all the lambdas for the gaussians
    # j is an index of a particular lambda
    q = (gauss_prob_density(xi, mean[j], covariance[j]) * lam[j])  
    return q


def m_step(qs, xs, k):
    N = len(x)
    lams = np.mean(qs, axis=0)

    means = []
    for y in range(k):
        mu_y = 0
        d = 0
        for i in range(N):
            mu_y += qs[i][y] * x[i]
            d += qs[i][y]
        means.append(mu_y / d)

    covariance = []
    for y in range(k):
        num = 0
        d = 0
        for i in range(N):
            vect = np.array(xs[i] - means[y])
            num += qs[i][y] * np.outer(vect, vect.T)
            d += qs[i][y]

        cov_elem = np.divide(num, d)
        covariance.append(cov_elem)

    return means, covariance, lams


def compute_objective(mean, covariance, lam, q, x, k):
    out = 0
    for i in range(len(x)):
        temp_out = 0
        for y in range(k):
            temp_out += q[i][y] * log(gauss_prob_density(x[i], mean[y], covariance[y]) / q[i][y])
        out += temp_out
    return out


def gauss_prob_density(x, mean, covariance):
    print(is_pos_def(covariance))
    return multivariate_normal.pdf(x, mean, covariance)



def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) >= 0) and np.all(x-x.T==0)


def gmm(x, k):
    covariance = [np.eye(len(x[0])) for _ in range(k)]
    mean = [[random.uniform(-3,3) for _ in range(len(x[0]))] for _ in range(k)]
    lam = np.random.dirichlet(np.ones(k), size=1)[0]
    iters = 0

    prev_cov = []
    prev_mean = []
    prev_lam = []
    prev_qs = []

    while True:
        print("Iteration ", iters)
        iters += 1
        
        # e_step
        qs = []
        for xi in x:
            d = (sum([gauss_prob_density(xi, mean[l], covariance[l]) * lam[l] for l in range(k)])) 
            qs.append([e_step(xi, mean, covariance, lam, j) / d for j in range(k)])


        # m_step
        mean, covariance, lam = m_step(qs,x,k)
      
        # keep track of previous iteration
        if  np.array_equal(qs, prev_qs) or np.array_equal(mean, prev_mean) or np.array_equal(covariance, prev_cov) or np.array_equal(lam, prev_lam):
            break
        
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
            mean, covariance, lam, qs = gmm(x, k)
            objectives.append(compute_objective(mean, covariance, lam, qs, x, k))

        print("k = %d \t mean = %f variance = %f" % (k, np.mean(objectives), np.var(objectives)))
