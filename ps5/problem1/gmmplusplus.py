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
            if q[i][y] > 1e-5 and gauss_prob_density(x[i], mean[y], covariance[y]) > 1e-5:
               temp_out += q[i][y] * log(gauss_prob_density(x[i], mean[y], covariance[y]) / q[i][y])
        out += temp_out
    return out


def gauss_prob_density(x, mean, covariance):
    try:
        return multivariate_normal.pdf(x, mean, covariance)
    except:
        covariance = covariance + np.diag([1e-8 for _ in range(len(covariance))])
        return multivariate_normal.pdf(x, mean, covariance)


def gmm(x, k):
    covariance = [np.eye(len(x[0])) for _ in range(k)]
    lam = np.random.dirichlet(np.ones(k), size=1)[0]
    mean = []
    mean.append(x[random.randint(0, len(x))])
    
    for remaining in range(k - 1):
        dist = []
        for i in range(np.array(x).shape[0]):
            point = x[i]
            d = np.inf

            for j in range(len(mean)):
                temp_dist = euclidean(point, mean[j])
                d = min(d, temp_dist)
            dist.append(d)
        dist = np.array(dist)
        next_mean = x[np.argmax(dist), :]
        mean.append(next_mean)

    prev_qs = [] 
    prev_loglike = 0
    while True:
                
        # e_step
        qs = []
        for xi in x:
            d = (sum([gauss_prob_density(xi, mean[l], covariance[l]) * lam[l] for l in range(k)])) 
            qs.append([e_step(xi, mean, covariance, lam, j) / d for j in range(k)])


        # keep track of previous iteration
        # if len(prev_qs) != 0:
            #print(np.allclose(qs, prev_qs, rtol=1e-03, atol=1e-03) or np.allclose(prev_qs, qs, rtol=1e-03, atol=1e-03))
        if len(prev_qs) != 0 and (np.allclose(qs, prev_qs, rtol=1e-03, atol=1e-03) or np.allclose(prev_qs, qs, rtol=1e-03, atol=1e-03)):
            #print('end because of prev_qs')
            break
        prev_qs = copy.deepcopy(qs)

        # m_step
        mean, covariance, lam = m_step(qs,x,k)
      
        loglike = compute_objective(mean, covariance, lam, qs, x, k)
        if abs(loglike - prev_loglike) < 1e-4:
            #print('end because of prev_loglike')
            break  
        prev_loglike = loglike

    return mean, covariance, lam, qs
    

if __name__ == "__main__":
    x, y = import_data('leaf.data')
    x = normalize_data(x) 
    
    ks = [12, 18, 24, 36, 42]
    for k in ks:
        objectives = [] 
        for i in range(20):            
            #print("random initialization %d for k = %d" % (i, k))
            mean, covariance, lam, qs = gmm(x, k)
            objectives.append(compute_objective(mean, covariance, lam, qs, x, k))

        print("k = %d \t mean = %f variance = %f" % (k, np.mean(objectives), np.var(objectives)))
