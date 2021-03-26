import numpy as np
import matplotlib.pyplot as plt
import math


from thes_graphics.heat_map_plot.plot_heat_map import plot_heat_map


num_values = 9
num_values = math.pow(math.ceil(math.sqrt(num_values)), 2)
arr = np.linspace(-10, 20, num_values)
reshape_len = int(math.sqrt(num_values))
arr_reshaped = np.reshape(arr, (reshape_len, reshape_len))

fig = plt.figure(1)
plot_heat_map(
    fig=fig,
    prior_skill_dist=((-5, -5), (5, 5)),
    heat_values=arr_reshaped,
    log=True
)
