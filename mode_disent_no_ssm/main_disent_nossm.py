import os
import argparse
import torch
import gym
from pprint import pprint
from datetime import datetime
from easydict import EasyDict as edict

from mode_disent.env_wrappers.rlkit_wrapper import NormalizedBoxEnvForPytorch
from mode_disent_no_ssm.agent import DisentTrainerNoSSM
from mode_disent_no_ssm.utils.parse_args import parse_args, yaml_save
from mode_disent_no_ssm.utils.skill_policy_wrapper import DiaynSkillPolicyWrapper

def run():
    args = parse_args()
    args.run_hp = args.copy()

    dir_name = args.env_info.env_id
    base_dir = os.path.join('logs', dir_name)
    if args.log_folder is not None:
        args.log_dir = os.path.join(base_dir, args.log_folder)
    else:
        args.log_dir = os.path.join(base_dir, args.run_id)
    args.params_for_testing = args.copy()

    args.device = args.device if torch.cuda.is_available() else "cpu"

    args.env = NormalizedBoxEnvForPytorch(
        gym_id=args.env_info.env_id,
        action_repeat=args.env_info.action_repeat,
        obs_type='state',
        normalize_states=True
    )

    skill_policy_object = torch.load(args.skill_policy_path)['evaluation/policy']
    args.skill_policy = DiaynSkillPolicyWrapper(skill_policy=skill_policy_object)
    args.mode_latent_model = torch.load(args.mode_model_path) \
        if args.mode_model_path is not None else None

    args.run_id = f'mode_disent{args.seed}-{datetime.now().strftime("%Y%m%d-%H%M")}'
    if args.run_comment is not None:
        args.run_id += str(args.run_comment)

    args.pop('run_comment')
    args.pop('skill_policy_path')
    args.pop('log_folder')
    args.pop('mode_model_path')
    args.pop('env_info')

    pprint(args)
    agent = DisentTrainerNoSSM(**args)
    agent.run_training()


if __name__ == "__main__":
    run()
