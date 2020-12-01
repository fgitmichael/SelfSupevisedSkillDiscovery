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

    algo_file_name, algo_epoch = get_numbered_file(algo_name, base_dir=base_dir)
    _, algo_file_extension = os.path.splitext(algo_file_name)
    assert algo_file_extension == file_extension

    config = torch.load(os.path.join(base_dir, config_name + file_extension))
    start_algo = algo_creator_fun(
        config=config,
        config_path_name=None,
    )
    start_algo.load(
        file_name=algo_file_name,
        base_dir=base_dir,
    )

    return start_algo


def get_numbered_file(
        base_name,
        base_dir='.',
):
    """
    Gets file_name and number of files with structure: base_name[...]number[...]
    """
    with os.scandir(base_dir) as dir:
        algo_files = []
        for dir_entry in dir:
            path = dir_entry.path
            _, file_name = os.path.split(path)
            if file_name.startswith(base_name):
                algo_files.append(file_name)
        assert len(algo_files) == 1
        digit_str = ''.join(filter(lambda  i: i.isdigit(), algo_files[0]))
        number = int(digit_str)

    return algo_files[0], number
