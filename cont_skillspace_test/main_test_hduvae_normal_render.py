# Working directory
import torch
import argparse

from cont_skillspace_test.visualization_fun.env_viz_render import \
    EnvVisualizationRenderHduvae


import rlkit.torch.pytorch_util as ptu


ptu.set_gpu_mode(False)

parser = argparse.ArgumentParser()
parser.add_argument('--epoch',
                    type=int,
                    default=100,
                    help="epoch to test",
                    )
args = parser.parse_args()
ptu.set_gpu_mode(True)
epoch = args.epoch
extension = ".pkl"
policy_net_name = "policy_net_epoch{}".format(epoch) + extension
env_name = "env" + extension
skill_selector_name = "skill_selector" + extension
env = torch.load(env_name)
policy = torch.load(policy_net_name, map_location=ptu.device)
skill_selector = torch.load(skill_selector_name, map_location=ptu.device)

env_viz = EnvVisualizationRenderHduvae(
    env=env,
    policy=policy,
    skill_selector=skill_selector,
    seq_len=100,
    render_dt=0.000001,
)
env_viz.run()
