import numpy as np
import csv
import copy


def pca(data):
    M = np.mean(data.T, axis=1)
    C = data - M
    WWT = np.cov(C.T)
    
    eig_vals, eig_vect = np.linalg.eig(WWT)
    print(sorted(eig_vals, reverse=True))

    return sorted(eig_vals, reverse=True)
    # from sklearn.preprocessing import scale, normalize
    # scaled_data = preprocessing.scale(data)
    # print(scaled_data)
    # mus = []
    # p = len(data)
    # data = data.T
    # for i in data: 
    #     mus.append(sum(i) / float(p))

    # data = data.T
    # w = copy.deepcopy(data)
    # for i in range(len(w)):
    #     w[i] = np.subtract(w[i], mus)

    
    # wwt = np.matmul(w, w.T)
    # eig_val, eig_vect = np.linalg.eig(wwt)
    # print(sorted(eig_val.real,reverse=True))
    # print(eig_vect)
    # wwt = np.cov(w)
    # eigs, normeigs = np.linalg.eig(wwt)
    # print(eigs)
    # print(normeigs)

def svm(data, eig_vals):
    

def main():
    all_data = np.loadtxt(fname="sonar_train.data", delimiter=',')
    labels = all_data[:,-1]
    data = all_data[: , 0:-1]

    eig_vals = pca(data)[0:6]


if __name__ == "__main__":
    main()