import os
import torch

from latent_with_splitseqs.base.tb_logging_base import PostEpochDiagnoWritingBase

from diayn_original_tb.algo.algo_diayn_tb import DIAYNTorchOnlineRLAlgorithmTb

algo_logging_dir_name = "algo_continuation_logging"
algo_name = "algo"
config_name = "config"
file_extension = ".pkl"


class ConfigSaver(PostEpochDiagnoWritingBase):

    def __init__(self,
                 config,
                 diagno_writer,
                 ):
        super().__init__(
            diagnostic_writer=diagno_writer,
        )
        self.config = config
        # Create own directory for algo saving
        self.algo_logging_dir = _create_algo_logging_dir(
            diagno_writer=diagno_writer)

        self.config_written = False

    def __call__(self, *args, epoch, **kwargs):
        if not self.config_written:
            save_path = os.path.join(
                self.algo_logging_dir,
                config_name + file_extension,
            )
            torch.save(
                obj=self.config,
                f=save_path,
            )
            self.config_written = True


def _create_algo_logging_dir(diagno_writer):
    run_dir = diagno_writer.writer.run_dir
    algo_logging_dir = os.path.join(run_dir, algo_logging_dir_name)
    if not os.path.exists(algo_logging_dir):
        os.makedirs(algo_logging_dir)
    return algo_logging_dir


def save_algo(self, *args, epoch, **kwargs):
    """
    Post Epoch Function for saving algo files
    """
    assert isinstance(self, DIAYNTorchOnlineRLAlgorithmTb)
    # Create own directory for algo saving
    algo_logging_dir = _create_algo_logging_dir(diagno_writer=self.diagnostic_writer)

    # Delete last algo
    with os.scandir(algo_logging_dir) as dir:
        for dir_entry in dir:
            path = dir_entry.path
            assert isinstance(path, str)
            if path.startswith(algo_name):
                os.remove(path)

    # Save new algo
    if epoch > 0:
        self.save(
            file_name=algo_name + "_epoch{}".format(epoch) + file_extension,
            base_dir=algo_logging_dir,
        )
