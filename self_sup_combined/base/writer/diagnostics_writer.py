import torch
import os
import sys
import gym
from shutil import copyfile

from self_supervised.utils.writer import MyWriterWithActivation


class DiagnosticsWriter:

    def __init__(self,
                 writer: MyWriterWithActivation,
                 log_interval=None
                 ):
        self.log_interval = log_interval
        self._diagnostics = {}

        self.writer = writer
        self.copy_main_script()

    def is_log(self, step, log_interval=None) -> bool:
        if log_interval is None:
            log_interval = self.log_interval

        if step % log_interval == 0:
            return True

    def save_object(self,
                    obj,
                    save_name: str,
                    epoch: int,
                    log_interval: int
                    ):
        if self.is_log(epoch, log_interval) and epoch > 0:
            save_name = "{}_epoch{}.pkl".format(save_name, epoch)
            save_path = os.path.join(self.writer.model_dir, save_name)
            torch.save(obj, save_path)

    def save_env(self,
                 env: gym.Env):
        save_path = os.path.join(self.writer.model_dir, "env.pkl")
        torch.save(env, save_path)

    def copy_main_script(self):
        script_path = sys.argv[0]
        save_path = os.path.join(self.writer.summary_dir, 'main_script.py')
        copyfile(script_path, save_path)
