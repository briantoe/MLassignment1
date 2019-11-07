import numpy as np
import copy
import random
import naivebayes
np.set_printoptions(threshold=1000)


def pca(data):
    M = np.mean(data.T, axis=1)
    C = data - M
    WWT = np.cov(C.T)
    
    eig_vals, eig_vect = np.linalg.eig(WWT)
    # print(eig_vals, eig_vect)
    # print(sorted(eig_vals, reverse=True))
    

    return (sorted(eig_vals, reverse=True), [x for _,x in sorted(zip(eig_vals,eig_vect), reverse=True)])
   

def full_pca(k, suppressOutput=True):
    np.set_printoptions(threshold=1000)
    all_data = np.loadtxt(fname="../sonar_train.data", delimiter=',')
    labels = all_data[:,-1]
    data = all_data[: , 0:-1]

    eig_vals, eig_vects = pca(data)
 
    # only care for first 6 eigenvalues 
    if suppressOutput == False:
        print("First k = %d eigenvalues of the data covariance matrix: " % (k))
        print(eig_vals[0:k])

    return eig_vals[0:k], eig_vects[0:k]

def demonstration():
    # adjust these parameters as needed
    k = 2
    s = 10

    eig_vals, eig_vects = full_pca(k=k, suppressOutput=False)
    i = 0
    for vect in eig_vects:
        eig_vects[i] = [x ** 2 for x in vect]
        i += 1
    pi = np.mean(np.array(eig_vects), axis=0)
    print(pi)
    print('\nRandomly sampling s = %d columns from pi' %(s))
    for _ in range(s):
        r = random.randint(0, len(pi) - 1)
        print(pi[r])
    print('\n')


def nb():
    train_data = np.loadtxt(fname="../sonar_train.data", delimiter=',')
    train_labels = train_data[:,-1]
    train_data = train_data[: , 0:-1]

    test_data = np.loadtxt(fname="../sonar_test.data", delimiter=',')
    test_labels = test_data[:,-1]
    test_data = test_data[: , 0:-1]

    ks = [i for i in range(1,11)]
    ss = [i for i in range(1, 21)]
    for k in ks:
        eig_vals, eig_vects = full_pca(k=k)   
        i = 0
        for vect in eig_vects:
             eig_vects[i] = [x ** 2 for x in vect]
             i += 1
        pi = np.mean(np.array(eig_vects), axis=0)
        for s in ss:
            for _ in range(1):
                uniquecols = []
                for _ in range(s):
                    r = random.randint(0, len(pi) - 1)
                    uniquecols.append(r)
                temp = np.array(uniquecols)
                uniquecols = np.unique(temp)
                mapping_vect = []
                for col in uniquecols:
                    mapping_vect.append(pi[col])
                mapping_vect = np.array(mapping_vect)
                print(mapping_vect)
                # apply mapping to the dataset
                ds = np.dot(mapping_vect, train_data.T)
                print(ds)
                # now do the naiive bayes thing
                # classifier = naivebayes.train( 
            
            

if __name__ == "__main__":
    demonstration()
    nb()



