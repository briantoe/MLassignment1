import numpy as np
import csv


all_data = np.loadtxt(fname="sonar_train.data", delimiter=',')



labels = all_data[:,-1]
data = all_data[: , 0:-1]



def pca(data):
    from sklearn import preprocessing
    # mus = []
    # data = data.T
    # for i in data:
    #     mus.append(sum(i) / len(i))

    # data = data.T
    # for i in range(len(data)):
    #     data[i] = np.subtract(data[i], mus)
    
    scaled_data = preprocessing.scale(data)
    
        

pca(data)