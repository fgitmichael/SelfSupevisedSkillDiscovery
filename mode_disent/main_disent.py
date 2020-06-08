import os
import argparse
import torch
import gym
from pprint import pprint
from datetime import datetime
from easydict import EasyDict as edict

from code_slac.env.ordinary_env import OrdinaryEnvForPytorch
from mode_disent.agent import DisentAgent
from mode_disent.utils.utils import parse_args


def run():
    # Parse arguments from command line
    args = parse_args()

    args.device = args.device if torch.cuda.is_available() else "cpu"
    args.env = OrdinaryEnvForPytorch(args.env_id)

    args.skill_policy = torch.load(args.skill_policy_path)['evaluation/policy']
    args.dyn_latent = torch.load(args.dynamics_model_path)\
        if args.dynamics_model_path is not None else None

    args.run_id = f'mode_disent{args.seed}-{datetime.now().strftime("%Y%m%d-%H%M")}'
    if args.run_comment is not None:
        args.run_id += str(args.run_comment)

    dir_name = args.domain
    base_dir = os.path.join('logs', dir_name)
    if args.log_folder is not None:
        args.log_dir = os.path.join(base_dir, args.log_folder)
    else:
        args.log_dir = os.path.join(base_dir, args.run_id)

    args.pop('env_id')
    args.pop('run_comment')
    args.pop('domain')
    args.pop('skill_policy_path')
    args.pop('log_folder')
    args.pop('dynamics_model_path')
    agent = DisentAgent(**args).run()


# python main_disent.py --config ./config/config.json
if __name__ == "__main__":
    run()
