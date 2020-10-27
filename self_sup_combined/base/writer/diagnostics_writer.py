import torch
import os
import sys
import gym
from shutil import copyfile
from easydict import EasyDict as edict

from self_supervised.utils.writer import MyWriterWithActivation

from mode_disent_no_ssm.utils.parse_args import yaml_save, json_save


class DiagnosticsWriter:

    def __init__(self,
                 writer: MyWriterWithActivation,
                 config: edict = None,
                 config_path_name: str = None,
                 test_script_path_name: str = None,
                 log_interval=None
                 ):
        self.log_interval = log_interval
        self._diagnostics = {}
        self.writer = writer

        self.copy_main_script()
        self.save_hparams(config)
        self.copy_config(config_path_name)
        self.create_test_script_symlink(test_script_path_name)

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

    def copy_config(self, config_path_name):
        if config_path_name is not None:
            name = os.path.basename(config_path_name)
            save_path = os.path.join(self.writer.summary_dir, name)
            copyfile(config_path_name, save_path)

        else:
            print("No config file copied")

    def create_test_script_symlink(self, test_script_path_name):
        if test_script_path_name is not None:
            if isinstance(test_script_path_name, list) or \
                isinstance(test_script_path_name, tuple):
                for path_name in test_script_path_name:
                    self._create_test_script_symlin(
                        path_name,
                        link_name=os.path.basename(path_name)
                    )
            else:
                self._create_test_script_symlin(test_script_path_name)

        else:
            print("No testscript symlink created")

    def _create_test_script_symlin(self, test_script_path_name, link_name=None):
        if link_name is None:
            link_name = "test_script.py"

        sym_link_path = os.path.join(self.writer.model_dir, link_name)
        if not os.path.exists(sym_link_path):
            os.symlink(
                src=test_script_path_name,
                dst=sym_link_path
            )

    def save_hparams(self, hparams: edict):
        if hparams is not None:
            save_path = os.path.join(self.writer.summary_dir, 'hparams.yaml')
            json_save(
                path_name=save_path,
                file=hparams,
            )

        else:
            print("No config file used")
