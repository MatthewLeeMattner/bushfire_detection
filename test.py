import matplotlib.pyplot as plt
import numpy as np
import mpl_toolkits.mplot3d.axes3d as axes3d

np.random.seed(1)
fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
ax.tick_params(axis='x', length=20)
#fig.set_size_inches(18.5, 3.5, forward=True)
N = 100
X, Y = np.meshgrid(np.arange(N), np.arange(500))
heights = np.random.randn(500, N)
ax.plot_surface(X, Y, heights, cmap=plt.get_cmap('jet'))
plt.show()