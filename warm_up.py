import numpy as np
import matplotlib.pyplot as plt


def f_mega(t):
    return np.maximum((t-1)**2, (t-2)**2)


t3 = np.arange(-3.0, 6.0, 0.02)
plt.plot(t3, f_mega(t3), 'r-')
plt.axis([-6, 8, -1, 10])
plt.show()
