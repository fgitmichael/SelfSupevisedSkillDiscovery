import argparse
import os

from latent_with_splitseqs.utils.load_algo import load_algo
from latent_with_splitseqs.main_all_in_one_horizon_step_collector import create_experiment
from latent_with_splitseqs.post_epoch_funcs.algo_saving import algo_logging_dir_name


import rlkit.torch.pytorch_util as ptu

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu',
                        type=int,
                        default=1,
                        help="1: gpu, 0: cpu"
                        )
    args = parser.parse_args()
    gpu = bool(args.gpu)
    ptu.set_gpu_mode(gpu)
    algo_instance = load_algo(
        algo_creator_fun=create_experiment,
        base_dir=os.path.join('..', algo_logging_dir_name)
    )
    algo_instance.to(ptu.device)
    algo_instance.train(start_epoch=algo_instance.epoch_cnt + 1)
