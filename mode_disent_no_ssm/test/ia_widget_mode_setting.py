import torch
from typing import Union
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

from mode_disent.test.action_sampler import ActionSamplerWithActionModel
from gym import Env
from matplotlib.widgets import Slider, Button


class IaVisualizationNoSSM:

    def __init__(self,
                 fig,
                 change_mode_fun: Union[ActionSamplerWithActionModel.set_mode,
                                        ActionSamplerWithActionModel.set_mode_next],
                 reset_env_fun: Env.reset,
                 update_rate: int):
        #self.fig, self.ax = plt.subplots()
        self.fig = fig

        width = 3
        self.v = plt.axvline(0.6, ymin=-5, ymax=5)
        self.h = plt.axhline(0.1, xmin=-5, xmax=5)

        plt.subplots_adjust(left=0.25, bottom=0.5)

        bottom = 0.05
        widget_height = 0.05
        self.ax_slid1 = plt.axes([0.25, bottom + 3 * widget_height, 0.65, 0.03])
        self.ax_slid2 = plt.axes([0.25, bottom + 2 * widget_height, 0.65, 0.03])
        self.ax_start_button = plt.axes([0.25, bottom + widget_height, 0.65, 0.03])
        self.ax_stop_button = plt.axes([0.25, bottom, 0.65, 0.03])

        self.slid1 = Slider(self.ax_slid1, 'x_pos', -width, width,
                            valinit=0.5, valstep=0.01)
        self.slid2 = Slider(self.ax_slid2, 'y_pos', -width, width,
                            valinit=0.5, valstep=0.01)
        self.start_button = Button(self.ax_start_button, label='Start')
        self.stop_button = Button(self.ax_stop_button, label='Stop')

        self.slid1.on_changed(self._slider_callback)
        self.slid2.on_changed(self._slider_callback)
        self.start_button.on_clicked(self._start_button_callback)
        self.stop_button.on_clicked(self._stop_button_callback)

        self.change_mode_fun = change_mode_fun
        self.reset_env_fun = reset_env_fun

        plt.interactive(False)
        plt.pause(0.05)

        self.steps = 0
        self.update_rate = update_rate

        self._should_start = False
        self._should_stop = False

    def _start_button_callback(self, a):
        print('start button')
        x_pos = self.slid1.val
        y_pos = self.slid2.val
        mode = torch.tensor([x_pos, y_pos]).unsqueeze(0)
        self.change_mode_fun(mode=mode)

        self._should_start = True

        print('updated_mode to ' + str(mode))

    def _stop_button_callback(self, a):
        print('stop button')
        self._should_stop = True

    def _slider_callback(self, a):
        x_pos = self.slid1.val
        y_pos = self.slid2.val
        self.v.set_xdata(x_pos)
        self.h.set_ydata(y_pos)

    def update_plot(self):
        if self.steps % self.update_rate == 0:
            plt.pause(0.001)
        self.steps += 1

    def is_should_stop(self):
        if self._should_stop is True:
            return_val = True

        else:
            return_val = False

        return return_val

    def is_should_start(self):
        if self._should_start is True:
            self._should_start = False
            self._should_stop = False

            return_val = True

        else:
            return_val = False

        return return_val
