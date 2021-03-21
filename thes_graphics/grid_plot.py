import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import tikzplotlib

from cont_skillspace_test.grid_rollout.grid_rollouter \
    import create_twod_grid


show = False
if not show:
    matplotlib.use("pgf")
    matplotlib.rcParams.update({
        "pgf.texsystem": "pdflatex",
        'font.family': 'serif',
        'text.usetex': True,
        'pgf.rcfonts': False,
    })
matplotlib.rcParams['text.usetex'] = True

low = np.array([-2.5, -2.5])
high = np.array([2.5, 2.5])

grid = create_twod_grid(
    low=low,
    high=high,
    num_points=9,
    matrix_form=True
)

fig = plt.figure()
plt.scatter(grid[:, 0], grid[:, 1])
x_ticks = [-2.5, 0, 2.5]
x_labels = [r'$a_x$', 0, r'$b_x$']
y_ticks = [-2.5, 0, 2.5]
y_labels = [r'$a_y$', 0, r'$b_y$']

plt.xticks(
    ticks=x_ticks,
    labels=x_labels,
)
plt.yticks(
    ticks=y_ticks,
    labels=y_labels,
)

plt.grid(True)

if show:
    plt.show()

else:
    size_inches = 2.5
    fig.set_size_inches(w=size_inches, h=size_inches)

    print("path: ")
    path = input()
    plt.savefig(path)


    print("path: ")
    path = input()
    tikzplotlib.clean_figure()
    tikzplotlib.save(path)

print('End')