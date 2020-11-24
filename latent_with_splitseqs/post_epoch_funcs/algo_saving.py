import os
import numpy as np
import torch

from latent_with_splitseqs.base.tb_logging_base import PostEpochDiagnoWritingBase
import copy

from diayn_original_tb.algo.algo_diayn_tb import DIAYNTorchOnlineRLAlgorithmTb


config_file_name = "hparams.pkl"
algo_logging_dir_name = "algo_continuation_logging"
obj_save_names = dict(
    replay_buffer="replay_buffer",
    expl_step_collector="expl_step_collector",
    eval_path_collector="eval_path_collector",
    seq_eval_collector="seq_eval_collector",
    eval_policy="eval_policy",
    df="df",
    expl_env="expl_env",
    eval_env="eval_env",
    trainer="trainer",
    diagno_writer="diagno_writer",
)
algo_class_name = "algo_class"
file_extension = ".pkl"


class AlgoLogger(PostEpochDiagnoWritingBase):

    def __init__(self,
                 replay_buffer,
                 expl_step_collector,
                 eval_path_collector,
                 seq_eval_collector,
                 eval_policy,
                 df,
                 config,
                 expl_env,
                 eval_env,
                 trainer,
                 diagno_writer,
                 ):
        super(AlgoLogger, self).__init__(
            diagnostic_writer=diagno_writer,
        )
        self.config = config
        self.to_log = {
            obj_save_names['replay_buffer']: replay_buffer,
            obj_save_names['diagno_writer']: diagno_writer,
            obj_save_names['expl_step_collector']: expl_step_collector,
            obj_save_names['eval_path_collector']: eval_path_collector,
            obj_save_names['seq_eval_collector']: seq_eval_collector,
            obj_save_names['eval_policy']: eval_policy,
            obj_save_names['df']: df,
            obj_save_names['expl_env']: expl_env,
            obj_save_names['eval_env']: eval_env,
            obj_save_names['trainer']: trainer,
        }

        # Create own directory for algo saving
        run_dir = self.diagno_writer.writer.run_dir
        self.algo_logging_dir = os.path.join(run_dir, algo_logging_dir_name)
        if not os.path.exists(self.algo_logging_dir):
            os.makedirs(self.algo_logging_dir)

        # Pickle config
        file_name = os.path.join(self.algo_logging_dir, config_file_name)
        #torch.save(config, file_name + file_extension)
        np.save(file_name, config)

    def __call__(self, *args, epoch, **kwargs):
        for key, obj in self.to_log.items():
            torch.save(
                obj=obj,
                f=os.path.join(self.algo_logging_dir,
                               key + file_extension)
            )


def load_algo(
        algo_creator_func,
        base_dir='.',
) -> DIAYNTorchOnlineRLAlgorithmTb:
    base_dir = base_dir

    # Load objects
    objs = {}
    for name in obj_save_names.values():
        objs[name] = torch.load(
            os.path.join(base_dir, name + file_extension)
        )

    # Load algo class
    algo_class = torch.load(
        os.path.join(base_dir, algo_class_name + file_extension)
    )

    # Load config
    config_obj = torch.load(
        os.path.join(base_dir, config_file_name + file_extension)
    )

    algo_instance = algo_creator_func(
        **objs,
        config=config_obj,
        algo_class_in=algo_class
    )

    return algo_instance
