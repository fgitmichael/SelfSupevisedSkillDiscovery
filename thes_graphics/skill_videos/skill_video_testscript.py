import torch
import os
import numpy as np

from cont_skillspace_test.utils.load_env import load_env

from thes_graphics.skill_videos.video_rollouter import VideoGridRollouter
from thes_graphics.skill_videos.relevant_video_saver import RelevantTrajectoryVideoSaver
from thes_graphics.rollout_relevant_eval.max_plus_minus_x import extract_max_plus_minus_x
from thes_graphics.rollout_relevant_eval.max_abs_x import extract_max_abs_x

import rlkit.torch.pytorch_util as ptu

filedir = './files'
env = load_env(dir=filedir)
policy = torch.load(os.path.join(filedir, 'policy_net_epoch25000.pkl'))
config = torch.load(os.path.join(filedir, 'config.pkl'))
skill_dim = config['skill_dim']
grid_low = np.array([config['skill_prior']['uniform']['low']
                     for _ in range(skill_dim)])
grid_high = np.array([config['skill_prior']['uniform']['high']
                      for _ in range(skill_dim)])
horizon_len = 20
ptu.set_gpu_mode(config['gpu'])

rollouter = VideoGridRollouter(
    env=env,
    policy=policy,
    horizon_len=horizon_len,
)
video_saver = RelevantTrajectoryVideoSaver(
    test_rollouter=rollouter,
    extract_relevant_rollouts_fun=extract_max_abs_x,
    num_relevant_skills=2,
    path='./test_videos',
    save_name_prefix='testvideo',
)

video_saver(
    grid_low=grid_low,
    grid_high=grid_high,
    num_points=2,
)