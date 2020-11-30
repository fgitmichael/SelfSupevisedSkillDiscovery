import os

import torch

from diayn_original_tb.algo.algo_diayn_tb import DIAYNTorchOnlineRLAlgorithmTb
from latent_with_splitseqs.main_all_in_one_horizon_step_collector import create_experiment
from latent_with_splitseqs.post_epoch_funcs.algo_saving import config_name, file_extension, algo_name


def load_algo(
        algo_creator_fun: create_experiment,
        base_dir='.',
) -> DIAYNTorchOnlineRLAlgorithmTb:
    base_dir = base_dir

    config = torch.load(
        os.path.join(base_dir, config_name + file_extension)
    )
    start_algo = algo_creator_fun(
        config=config,
        config_path_name=None,
    )
    start_algo.load(
        file_name=algo_name + file_extension,
        base_dir=base_dir,
    )

    return start_algo