# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 21:10:31 2019

@author: puyag
"""
import numpy as np
import math

def retrieveData(filename):
    data = np.genfromtxt(filename, delimiter=',')
    x = data[:,1:]
    y = data[:,0]
    for i in range(len(y)):
        if y[i] == 0:
            y[i] = -1
    return x,y

def normalizeData(data):
    count = 0
    for col in zip(*data):
        col = (col - mean(col)) / std(col)
        data[:,count] = col
        count += 1

def mean(nums):
    return sum(nums)/(float(len(nums)))

def variance(nums):
    average = mean(nums)
    return sum([(x - average)**2 for x in nums]) / float(len(nums) - 1)

def std(nums):
    return math.sqrt(variance(nums))

def accuracy(data, labels, w, b):
    count = len(data)
    for i in range(len(data)):
        if(-labels[i] * (np.dot(w, data[i]) + b) >= 0):
            count -= 1
    return count

def logistic_function(x, w, b):
    return np.exp(np.dot(w, x) + b) / (1 + np.exp(np.dot(w, x) + b))

def gradient_ascent(data, labels, w, b):
    d = [((labels[i] + 1) / 2.0) - logistic_function(data[i], w, b) for i in range(len(data))]
    db = np.sum(d)
    dw = [np.dot(data[:,i], d) for i in range(len(data[0]))]
    return db, np.array(dw)

def logistic_regression(data, labels):
    w = np.zeros(len(data[0]))
    b = 0
    #while not converged(data, labels, w, b):
    for i in range(3):
        db, dw = gradient_ascent(data, labels, w, b)
        w += dw
        b += db
    return w, b

if __name__ == "__main__" :
    x_train,y_train = retrieveData('park_train.data')
    x_test,y_test = retrieveData('park_test.data')
    normalizeData(x_train)
    w, b = logistic_regression(x_train, y_train)
    print("learned w:", w)
    print("learned b:", b)
    normalizeData(x_test)
    print(accuracy(x_test, y_test, w, b)/len(x_test))