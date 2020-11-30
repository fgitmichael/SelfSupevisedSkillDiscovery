import torch
import os

from diayn_original_tb.algo.algo_diayn_tb import DIAYNTorchOnlineRLAlgorithmTb

from latent_with_splitseqs.utils.load_algo import load_algo
from latent_with_splitseqs.config.fun.get_algo import get_algo_with_post_epoch_funcs
from latent_with_splitseqs.main_all_in_one_horizon_step_collector import create_experiment
from latent_with_splitseqs.post_epoch_funcs.algo_saving import algo_logging_dir_name


import rlkit.torch.pytorch_util as ptu

if __name__ == "__main__":
    ptu.set_gpu_mode(True)
    algo_instance = load_algo(
        algo_creator_fun=create_experiment,
        base_dir=os.path.join('..', algo_logging_dir_name)
    )
    algo_instance.to(ptu.device)
    algo_instance.train(start_epoch=algo_instance.epoch_cnt + 1)
