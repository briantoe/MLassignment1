from mpl_toolkits import mplot3d

import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

xs = np.linspace(0.9, 1.07, 10)
zs = np.linspace(-2.0, 1.0, 10)
X, Z = np.meshgrid(xs, zs)
Y = X - 1


ax.plot_surface(X, Z, Y, color=(0,1,0,0.2),shade=False)


zdata = 1
xdata = 1
ydata = 1

ax.scatter(xdata,ydata,zdata)
ax.scatter(xdata,ydata,-zdata,c='Red')


plt.ylim((.95, 1.05))
plt.xlim((.95, 1.05))
ax.set_zlim(-2,1.0)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.show()