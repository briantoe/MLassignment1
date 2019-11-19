import numpy as np
from math import sqrt
import random
from scipy.spatial.distance import euclidean
import copy


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

def kmeans(x, k):
    means = [] # mus
    clusters = dict() # S
  
    
    # randomly assign positions for k centroids of clusters in len(x[0]) dimensional space
    for _ in range(k):
        means.append([random.uniform(-3,3) for _ in range(len(x[0]))])
        # clusters[i] = [] # bucket that contains all of the datapoints assigned to each cluster

    # keep track of previous means, if there's no change between the current means and previous means then the algorithm is converged
    previous_means = None
    while True:
        clusters = dict() # S 
        # initialize the buckets for S
        for i in range(k):
            clusters[i] = []

        for datapoint in x: # iterate through each datapoint
            dists = [] 
            for mean in means: # Want to calculate distances^2 for all means and store the minimum value
                dists.append(euclidean(datapoint, mean) ** 2)
            m = min(dists)
            # the index of the minimum distance can access the ith cluster (bucket) to put the datapoint into
            clusters[dists.index(m)].append(datapoint)


        for key in clusters: # iterate through all clusters (buckets)
            if len(clusters[key]) == 0: # if there's nothing in the bucket, ignore it
                continue
            else:   # find the mean of that bucket, and update the ith mu/mean to be the bucket's mean
                means[key] = np.mean(clusters[key], axis=0)
        
        if np.array_equal(means, previous_means): # if the update didn't change anything, k-means has converged
            break
        previous_means = copy.deepcopy(means)
    
    return means, clusters

def compute_objective(means, clusters):
    dist = 0
    for key in clusters:
        dist += sum([euclidean(x, means[key]) ** 2 for x in clusters[key]])
    
    return dist


if __name__ == "__main__":
    x, y = import_data('leaf.data')
    x = normalize_data(x)
    

    import time
    start_time = time.time()


    ks = [12, 18, 24, 36, 42]
    for k in ks:
        objectives = []
        for _ in range(20):
            means, clusters = kmeans(x, k)
            objectives.append(compute_objective(means,clusters))

        print("k = %d \t mean = %f variance = %f" % (k, np.mean(objectives), np.var(objectives)))
        


    print("--- %s minutes elapsed ---" % ((time.time() - start_time)/60.0))