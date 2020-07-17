import numpy as np
import matplotlib.pyplot as plt

grid = np.array([
    [0.75, 0.],
    [0., 0.],
    [0., 0.75],
    [-0.75, 0.],
    [0, -0.75],
    [1., 1.],
    [-1., -1.],
    [-1., 1],
    [1., -1],
    [0., -1.38]
], dtype=np.float)

grid_plot = grid.transpose()
_, axes = plt.subplots()
lim = [-3., 3.]
axes.set_xlim(lim)
axes.set_ylim(lim)
plt.scatter(grid_plot[0], grid_plot[1])
plt.show()

