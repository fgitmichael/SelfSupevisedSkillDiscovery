import torch
import matplotlib.pyplot as plt
from mode_disent.test.action_sampler import ActionSamplerWithActionModel
from matplotlib.widgets import Slider
import matplotlib
matplotlib.use('TkAgg')
import torch


class IaVisualization:

    def __init__(self,
                 ax,
                 fig,
                 change_mode_fun: ActionSamplerWithActionModel.change_mode_on_fly):
        #self.fig, self.ax = plt.subplots()
        self.ax = ax
        self.fig = fig

        width = 3
        self.ax.set_xlim(left=-width, right=width)
        self.ax.set_ylim(ymin=-width, ymax=width)
        self.v = plt.axvline(0.6, ymin=-5, ymax=5)
        self.h = plt.axhline(0.1, xmin=-5, xmax=5)

        plt.subplots_adjust(left=0.25, bottom=0.25)

        plt.subplots_adjust(left=0.25, bottom=0.25)
        self.ax_slid1 = plt.axes([0.25, 0.1, 0.65, 0.03])
        self.ax_slid2 = plt.axes([0.25, 0.15, 0.65, 0.03])
        self.slid1 = Slider(self.ax_slid1, 'x_pos', -width, width,
                            valinit=0.5, valstep=0.01)
        self.slid2 = Slider(self.ax_slid2, 'y_pos', -width, width,
                            valinit=0.5, valstep=0.01)

        self.slid1.on_changed(self.slider_callback)
        self.slid2.on_changed(self.slider_callback)

        self.change_mode_fun = change_mode_fun

        plt.interactive(False)
        plt.pause(0.05)

    def slider_callback(self, a):
        x_pos = self.slid1.val
        y_pos = self.slid2.val
        self.v.set_xdata(x_pos)
        self.h.set_ydata(y_pos)

        mode = torch.tensor([x_pos, y_pos]).unsqueeze(0)
        self.change_mode_fun(mode=mode)

        print('updated_mode to ' + str(mode))

        #fig.canvas.draw_idle()

    def update_plot(self):
        plt.pause(0.00005)

def change_mode_fun_test(mode):
    print('change_mode_fun')
    print(mode)


if __name__=='__main__':
    a = IaVisualization(1, 2, change_mode_fun_test)
    while True:
        plt.pause(0.05)
        pass

