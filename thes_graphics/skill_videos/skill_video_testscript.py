import torch
import numpy as np

from thes_graphics.skill_videos.video_rollouter import VideoGridRollouter
from thes_graphics.skill_videos.relevant_video_saver import RelevantTrajectoryVideoSaver
from thes_graphics.rollout_relevant_eval.max_plus_minus_x import extract_max_plus_minus_x
from thes_graphics.rollout_relevant_eval.max_abs_x import extract_max_abs_x


policy = torch.load('./env.pkl')
env = torch.load('./policy_net_epoch25000.pkl')
config = torch.load('./config.pkl')
horizon_len = 300

rollouter = VideoGridRollouter(
    env=env,
    policy=policy,
    horizon_len=horizon_len,
)
video_saver = RelevantTrajectoryVideoSaver(
    test_rollouter=rollouter,
    extract_relevant_rollouts_fun=extract_max_abs_x,
    num_relevant_skills=10,
    path='./test_videos',
    save_name_prefix='testvideo',
)
video_saver(
    grid_low=np.
)