import torch

from diayn_original_tb.algo.algo_diayn_tb import DIAYNTorchOnlineRLAlgorithmTb

import rlkit.torch.pytorch_util as ptu

from latent_with_splitseqs.post_epoch_funcs.algo_saving import load_algo
from latent_with_splitseqs.config.fun.get_algo import get_algo_with_post_epoch_funcs

if __name__ == "__main__":
    algo_instance = load_algo(
        algo_creator_func=get_algo_with_post_epoch_funcs,
    )
    algo_instance.to(ptu.device)
    algo_instance.train(start_epoch=algo_instance.epoch_cnt)
