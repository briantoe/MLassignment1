from numpy import *

from cvxopt import matrix, solvers

dataX, dataY = [], []
dataXExt = []

f = open("mystery.data", 'r')
log = open("perceptron.log", 'w')

for line in f.readlines():
    a = line.split(',')
    b = [float(i) for i in a]
    dataX.append(b[0 : 4])
    dataY.append(b[4])

dim, dataLen = 23, len(dataX)

def addFeature():  # feature map [x1,x2,x3,x4,x1^2,x2^2, ...., x3^4,x4^4,x1x2,x1x3,x1x4,....,x3x4] (16+6=22)
    for i in range(dataLen):
        xExt = dataX[i] + [x ** 2 for x in dataX[i]] + [x ** 3 for x in dataX[i]] + [x ** 4 for x in dataX[i]]
        for j in range(4):
            for k in range(j + 1, 4):
                xExt = xExt + [dataX[i][j] * dataX[i][k]]
        xExt = xExt
        dataXExt.append(xExt)

addFeature()


# find p
p = eye(dim)
p[dim - 1][dim - 1] = 0
p = matrix(p)

# find q
q = [0.0] * dim
q = matrix(q)

# find h
h = matrix([-1.0] * 1000)

# find g
g = []
for i in range(dataLen):
	e = [float(-e * dataY[i]) for e in dataXExt[i]]
	e.append(float(-dataY[i]))
	g.append(e)

g = array(g)
#g = g.T
g = matrix(g)

# call svm solver
sol=solvers.qp(p, q, g, h)

print(sol['x']) # w,b in a single vector, vector[0:22] is w, vector[22], which is the last one, means b