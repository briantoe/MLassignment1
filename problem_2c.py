from mpl_toolkits import mplot3d

import math
import numpy as np
import matplotlib.pyplot as plt

plus1_x = math.exp(-1)
plus1_y = math.exp(-1)
plus2_x = math.exp(1)
plus2_y = math.exp(1)

minus1_x = math.exp(-1)
minus1_y = math.exp(1)
minus2_x = math.exp(1)
minus2_y = math.exp(-1)



plt.plot(plus1_x,plus1_y, 'bo')
plt.plot(plus2_x,plus2_y, 'bo')
plt.plot(minus1_x,minus1_y, 'ro')
plt.plot(minus2_x,minus2_y,'ro')
plt.show()