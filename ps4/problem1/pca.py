import numpy as np
import copy
np.set_printoptions(threshold=1000)


def pca(data):
    M = np.mean(data.T, axis=1)
    C = data - M
    WWT = np.cov(C.T)
    
    eig_vals, eig_vect = np.linalg.eig(WWT)
    # print(eig_vals, eig_vect)
    # print(sorted(eig_vals, reverse=True))
    

    return (sorted(eig_vals, reverse=True), [x for _,x in sorted(zip(eig_vals,eig_vect), reverse=True)])
   


def main():
    np.set_printoptions(threshold=1000)
    all_data = np.loadtxt(fname="sonar_train.data", delimiter=',')
    labels = all_data[:,-1]
    data = all_data[: , 0:-1]

    eig_vals, eig_vects = pca(data)
 
    # only care for first 6 eigenvalues 
    print("First 6 eigenvalues of the data covariance matrix: ")
    print(eig_vals[0:6])

    


if __name__ == "__main__":
    main()