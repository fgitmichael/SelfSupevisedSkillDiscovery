import torch
import numpy as np
from typing import Union, List
import gym

def set_seeds(seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)

def set_env_seed(seed=0,
                 env=Union[gym.Env, List[gym.Env]]):
    if type(env) is not list:
        env.seed(seed)
    else:
        for env_ in env:
            env_.seed(seed)
