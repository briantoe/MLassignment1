# coding:utf-8
from math import *
import numpy as np


def reform_labels(labels):
    for i in range(len(labels)):
        if labels[i] == 2:
            labels[i] = -1

    return labels


def import_data(fname):
    all_data = np.loadtxt(fname=fname, delimiter=',')
    labels = all_data[:, -1]
    data = all_data[:, 0:-1]
    return data, reform_labels(labels), all_data

# Split the dataset by class values, returns a dictionary
def separate_by_class(dataset):
    separated = dict()
    for i in range(len(dataset)):
        vector = dataset[i]
        class_value = vector[-1]
        if (class_value not in separated):
            separated[class_value] = list()
        separated[class_value].append(vector)

    for key in separated:
        separated[key] = np.array(separated[key])
    return separated


def mean(ds):
    return sum(ds)/float(len(ds))


def std_dev(ds):
    avg = mean(ds)
    variance = sum([(x-avg)**2 for x in ds]) / float(len(ds)-1)
    return sqrt(variance)

def g_probability(x, mean, std):
    return 1/(sqrt(2*pi*std)) * exp(-((x-mean)**2)/(2*std**2))

def class_probability(summ, row): # gaussian prob
    total_rows = sum([summ[label][0][2] for label in summ])
    # total number of data points that are in the data set
    probs = dict()
    for label, class_summ in summ.items():
        # does number of occurances / total number of data points
        probs[label] = summ[label][0][2]/float(total_rows) 
        for i in range(len(class_summ)):
            mean, std, _ = class_summ[i]
            probs[label] *= g_probability(row[i], mean, std)

    return probs



def main():
    X_train, Y_train, train_dataset = import_data("sonar_train.data")
    X_test, Y_test, _ = import_data("sonar_test.data")

    sep = separate_by_class(train_dataset)
    # calculate the mean and std_dev for all columns in the dataset
    summary = dict()
    for key in sep:
        temp = [(mean(col), std_dev(col), len(col)) for col in sep[key].T]
        # len(col) is storing the number of occurrances a class has
        del(temp[-1])
        summary[key] = temp

    misclassifications = 0
    for x, y in zip(X_test,Y_test):
        probs = class_probability(summary, x)
        prediction = None
        if probs[1.0] > probs[-1.0]:
            prediction = 1
        elif probs[-1.0] > probs[1.0]:
            prediction = -1
        
        if prediction != y:
            misclassifications += 1
    
    accuracy = float(len(X_test) - misclassifications) / len(X_test) * 100.0
    print("Accuracy = %.2f%%" %(accuracy))
    
if __name__ == "__main__":
    main()
