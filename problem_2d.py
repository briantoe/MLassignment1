from mpl_toolkits import mplot3d

import math
import numpy as np
import matplotlib.pyplot as plt

plus1_x = -1 * math.sin(-1)
plus1_y = -1
plus2_x = 1 * math.sin(1)
plus2_y = 1

minus1_x = -1 * math.sin(1)
minus1_y = -1
minus2_x = 1 * math.sin(-1)
minus2_y = 1



plt.plot(plus1_x,plus1_y, 'bo')
plt.plot(plus2_x,plus2_y, 'bo')
plt.plot(minus1_x,minus1_y, 'ro')
plt.plot(minus2_x,minus2_y,'ro')

plt.axvline(x=0.0, color='k')
plt.show()