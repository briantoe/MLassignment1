from mpl_toolkits import mplot3d

import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure()
ax = plt.axes(projection='3d')


zdata = 1
xdata = 1
ydata = 1

ax.scatter(xdata,ydata,zdata)
ax.scatter(xdata,ydata,-zdata,c='Red')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.show()