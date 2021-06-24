import torch
import os
import numpy as np

from cont_skillspace_test.utils.load_env import load_env

from thes_graphics.skill_videos.relevant_video_saver import RelevantTrajectoryVideoSaver
from thes_graphics.rollout_relevant_eval.max_abs_x import extract_max_abs_x
from thes_graphics.rollout_relevant_eval.max_plus_minus_x import extract_max_plus_minus_x

from my_utils.rollout.frame_plus_obs_rollout import rollout as rollout_frame_plus_obs
from my_utils.rollout.grid_rollouter import GridRollouter

import rlkit.torch.pytorch_util as ptu


ptu.set_gpu_mode(False)

filedir = './files'
video_destination = './test_videos'
horizon_len = 30
num_points = 5
num_relevant_skills = 1

policy_net_name = os.path.join(filedir, 'policy_net_epoch25000.pkl')
config_name = os.path.join(filedir, 'config.pkl')

env = load_env(dir=filedir)
policy = torch.load(policy_net_name, map_location=ptu.device)
config = torch.load(config_name)
skill_dim = config['skill_dim']
grid_low = np.array([config['skill_prior']['uniform']['low']
                     for _ in range(skill_dim)])
grid_high = np.array([config['skill_prior']['uniform']['high']
                      for _ in range(skill_dim)])

rollouter = GridRollouter(
    env=env,
    policy=policy,
    horizon_len=horizon_len,
    rollout_fun=rollout_frame_plus_obs,
)
video_saver = RelevantTrajectoryVideoSaver(
    test_rollouter=rollouter,
    extract_relevant_rollouts_fun=extract_max_plus_minus_x,
    num_relevant_skills=num_relevant_skills,
    path=video_destination,
    save_name_prefix='testvideo',
)

video_saver(
    grid_low=grid_low,
    grid_high=grid_high,
    num_points=num_points,
)
