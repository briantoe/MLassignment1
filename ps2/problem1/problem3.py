import numpy as np
from scipy.spatial import distance as d
import math

np.set_printoptions(threshold=float('inf'))


def read_data(filename):
    data = []
    with open(filename) as file:
        for line in file:
            data_line = [float(item) for item in line.split(',')]
            data.append(data_line)
    data = np.array(data)
    labels = data[:, 0] # grab labels
    dim = len(data[0]) # dimension of the data
    data = data[:, 1:dim] # grab vectors

    return (data, labels)

def reform_labels(labels):
    for i in range(len(labels)):
        if labels[i] == 0:
            labels[i] = -1

    return labels


filename = "park_train.data"
raw_data = read_data(filename)
data = raw_data[0]
labels = raw_data[1]
labels = reform_labels(labels)

filename = "park_test.data"
raw_data = read_data(filename)
test_data = raw_data[0]
test_labels = raw_data[1]
test_labels = reform_labels(test_labels)


ks = [1,5,11,15,21]

for k in ks:
    distances = []
    for x2, test_label in zip(test_data, test_labels):
        distances_from_datapoint = []
        for x1, train_label in zip(data, labels):        
            dist = d.euclidean(x1,x2)
            distances_from_datapoint.append((dist,train_label))
        distances.append(distances_from_datapoint)


    # print(np.array(distances))
    for i in range(len(distances)):
        distances[i] = sorted(distances[i], key=lambda tup: tup[0])
      
    correct = 0
    for line, test_label in zip(distances, test_labels):
        classifier = 0
        for i in range(k):
            classifier += line[i][1]
        if np.sign(classifier) == np.sign(test_label): 
            correct += 1
        
    correct = float(correct / len(test_data)) * 100.0    
    print("k = %d, Accuracy = %.2f%%" % (k, correct))

