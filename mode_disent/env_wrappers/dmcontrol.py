import numpy as np
import gym
import matplotlib.pyplot as plt
import pylab as pl
from dm_control import suite

from code_slac.env.dm_control import DmControlEnvForPytorch

class MyDmControlEnvForPytorch(DmControlEnvForPytorch):

    def __init__(self, *args, **kwargs):
        super(MyDmControlEnvForPytorch, self).__init__(*args, **kwargs)

        self.render_fig = None
        self.render_ax = None
        self.rendered_img = None

    # render function does not work in the base class
    # (AttributeError: 'Environment' object has no attribute 'render')
    def render(self, mode='human'):
        pixels = self.get_pixels()

        if mode == 'rbg_array':
            return pixels

        elif mode == 'human':
            if self.render_fig is None:
                self.render_fig = plt.figure()
                self.render_ax = self.render_fig.add_subplot(111)

            self.show_img(pixels)

            return pixels

    def show_img(self, img):
        plt.interactive(False)

        if self.rendered_img is None:
            self.rendered_img = pl.imshow(img)
        else:
            self.rendered_img.set_data(img)
        pl.pause(.0001)
        pl.draw()

    def get_pixels(self):
        def get_physics(env):
            if hasattr(env, 'physics'):
                return env.physics
            else:
                return get_physics(env.wrapped_env())

        img = get_physics(self.env).render(**self.render_kwargs)

        return img
