import numpy as np
import gym
from dm_control import suite

from code_slac.env.dm_control import DmControlEnvForPytorch
from mode_disent.env_wrappers.dmcontrol import MyDmControlEnvForPytorch


def test1():
    domain_name = 'cheetah'
    task_name = 'run'

    env = suite.load(
        domain_name=domain_name, task_name=task_name)

    render_kwargs = dict(
        width=64,
        height=64,
        camera_id=0,
    )

    def get_physics(env):
        if hasattr(env, 'physics'):
            return env.physics
        else:
            return get_physics(env.wrapped_env())

    image = get_physics(env).render(**render_kwargs)
    env.step(env.action_space.sample())
    pass


def test2():
    env = DmControlEnvForPytorch(
        domain_name='cheetah',
        task_name='run',
    )

    env.reset()

    img = env.render()
    pass

def test3():
    env = MyDmControlEnvForPytorch(
        domain_name='cheetah',
        task_name='run'
    )

    env.reset()
    for step in range(1000):
        env.render(mode='human')
        time_step = env.step(env.action_space.sample())
        print(step)

def test4():
    env = gym.make('HalfCheetah-v2')

    env.reset()
    for step in range(1000):
        env.render()
        env.step(env.action_space.sample())
        print(step)


if __name__=='__main__':
    test1()
    #test2()
    #test3()
    #test4()



