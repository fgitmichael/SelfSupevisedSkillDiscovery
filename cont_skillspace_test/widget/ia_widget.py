import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.widgets import Slider, Button
matplotlib.use('TkAgg')


class IaVisualization:

    def __init__(self,
                 reset_fun,
                 set_next_skill_fun,
                 ):
        self.fig, self.ax = plt.subplots()

        self.reset_fun = reset_fun
        self.set_next_skill_fun = set_next_skill_fun

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
        self.reset_button = Button(self.ax_start_button, label='Reset')

        self.slid1.on_changed(self._slider_callback)
        self.slid2.on_changed(self._slider_callback)
        self.reset_button.on_clicked(self._reset_button_callback)

        plt.interactive(False)
        plt.pause(0.05)

        self._cursor_location = None

    def _reset_button_callback(self, a):
        self.reset_fun()
        self.set_next_skill_fun(cursor_location=self.cursor_location)

    def _slider_callback(self, a):
        x_pos = self.slid1.val
        y_pos = self.slid2.val
        self.v.set_xdata(x_pos)
        self.h.set_ydata(y_pos)

    @property
    def cursor_location(self) -> np.ndarray:
        x_pos = self.slid1.val
        y_pos = self.slid2.val
        self._cursor_location = np.array([x_pos, y_pos])
        return self._cursor_location

    def __del__(self):
        plt.close(self.fig)
