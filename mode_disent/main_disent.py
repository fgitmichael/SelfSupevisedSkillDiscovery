import os
import argparse
import torch
import gym
from pprint import pprint
from datetime import datetime
from easydict import EasyDict as edict
import json

from code_slac.env.ordinary_env import OrdinaryEnvForPytorch
from code_slac.env.dm_control import DmControlEnvForPytorch
from mode_disent.env_wrappers.rlkit_wrapper import NormalizedBoxEnvForPytorch
from mode_disent.agent import DisentAgent
from mode_disent.utils.utils import parse_args
# Note: Set path variable for Mujoco using ...
#       LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/michael/.mujoco/mujoco200/bin
#       Call using
#       python main_disent.py --config ./config/NAME_OF_CONFIG.json


def run():
    # Parse arguments from command line
    args = parse_args()

    dir_name = args.env_info.env_id
    base_dir = os.path.join('logs', dir_name)
    if args.log_folder is not None:
        args.log_dir = os.path.join(base_dir, args.log_folder)
    else:
        args.log_dir = os.path.join(base_dir, args.run_id)
    args.run_hp = args.copy()

    args.device = args.device if torch.cuda.is_available() else "cpu"

    obs_type = "state" if args.state_rep is True else "pixels"
    if args.env_info.env_type == 'normal':
        args.env = OrdinaryEnvForPytorch(args.env_info.env_id)
    elif args.env_info.env_type == 'dm_control':
        args.env = DmControlEnvForPytorch(
            domain_name=args.env_info.domain_name,
            task_name=args.env_info.task_name,
            action_repeat=args.env_info.action_repeat,
            obs_type=obs_type
        )
    elif args.env_info.env_type == 'normalized':
        args.env = NormalizedBoxEnvForPytorch(
            gym_id=args.env_info.env_id,
            action_repeat=args.env_info.action_repeat,
            obs_type=obs_type
        )
    else:
        raise ValueError('Env_type is not used in if else statements')

    args.skill_policy = torch.load(args.skill_policy_path)['evaluation/policy']

    if args.dual_training:
        assert args.dynamics_model_path is None
        assert args.mode_model_path is None
    args.dyn_latent = torch.load(args.dynamics_model_path)\
        if args.dynamics_model_path is not None else None
    args.mode_latent = torch.load(args.mode_model_path)\
        if args.mode_model_path is not None else None
    args.memory = torch.load(args.memory_path)\
        if args.memory_path is not None else None
    args.test_memory = torch.load(args.test_memory_path) \
        if args.test_memory_path is not None else None

    args.run_id = f'mode_disent{args.seed}-{datetime.now().strftime("%Y%m%d-%H%M")}'
    if args.run_comment is not None:
        args.run_id += str(args.run_comment)

    dual_training = args.dual_training

    args.pop('run_comment')
    args.pop('skill_policy_path')
    args.pop('log_folder')
    args.pop('dynamics_model_path')
    args.pop('mode_model_path')
    args.pop('memory_path')
    args.pop('test_memory_path')
    args.pop('env_info')
    args.pop('dual_training')

    pprint(args)
    agent = DisentAgent(**args)
    if not dual_training:
        agent.run()
    else:
        agent.run_dual_training()


if __name__ == "__main__":
    run()
