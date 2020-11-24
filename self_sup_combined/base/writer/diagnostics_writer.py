import abc
import torch
from torch.utils.tensorboard import SummaryWriter
import os
import sys
import gym
from shutil import copyfile
from easydict import EasyDict as edict

from self_supervised.utils.writer import MyWriterWithActivation

from mode_disent_no_ssm.utils.parse_args import yaml_save, json_save


class DiagnosticsWriterBase(object, metaclass=abc.ABCMeta):

    def __init__(self, writer: SummaryWriter):
        self.writer = writer

        self.save_str = 'diagno_writer.pkl'

    def save(self, base_dir='.'):
        save_path = os.path.join(base_dir, self.save_str)
        save_obj = self._save()
        torch.save(save_obj, save_path)

    def _save(self):
        writer_log_dir = self.writer.get_logdir()
        return dict(
            writer_log_dir=writer_log_dir,
        )

    def load(self, base_dir='.') -> dict:
        save_obj = torch.load(os.path.join(base_dir, self.save_str))
        self.writer = SummaryWriter(
            log_dir=save_obj['writer_log_dir']
        )
        return save_obj


class DiagnosticsWriter(DiagnosticsWriterBase):

    def __init__(self,
                 *args,
                 writer: MyWriterWithActivation,
                 config: edict = None,
                 config_path_name: str = None,
                 scripts_to_copy: str = None,
                 log_interval=None,
                 **kwargs
                 ):
        super(DiagnosticsWriter, self).__init__(*args, **kwargs)
        self.log_interval = log_interval
        self._diagnostics = {}
        self.writer = writer

        self.copy_main_script()
        self.save_hparams(config)
        self.copy_config(config_path_name)
        self.create_copy_script_symlinks(scripts_to_copy)

    def _save(self):
        save_obj = super(DiagnosticsWriter, self)._save()
        save_obj['log_interval'] = self.log_interval
        return save_obj

    def load(self, base_dir='.'):
        save_obj = super(DiagnosticsWriter, self).load(base_dir=base_dir)
        self.log_interval = save_obj['log_interval']
        return save_obj

    def is_log(self, step, log_interval=None) -> bool:
        if log_interval is None:
            log_interval = self.log_interval

        if step % log_interval == 0:
            return True

        else:
            return False

    def save_object_islog(self,
                          obj,
                          save_name: str,
                          epoch: int,
                          log_interval: int
                          ):
        if self.is_log(epoch, log_interval) and epoch > 0:
            self.save_object(
                obj=obj,
                save_name=save_name,
                epoch=epoch,
            )

    def save_object(
            self,
            obj,
            save_name: str,
            epoch: int = None,
    ):
        if epoch is None:
            save_name = "{}.pkl".format(save_name)
        else:
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

    def create_copy_script_symlinks(self, scripts_to_copy):
        if scripts_to_copy is not None:
            if isinstance(scripts_to_copy, list) or \
               isinstance(scripts_to_copy, tuple):
                for path_name in scripts_to_copy:
                    self._create_test_script_symlin(
                        path_name,
                        link_name=os.path.basename(path_name)
                    )
            else:
                self._create_test_script_symlin(scripts_to_copy)

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
            base_name = "hparams"

            # Json
            save_path = os.path.join(self.writer.summary_dir, base_name + '.json')
            json_save(
                path_name=save_path,
                file=hparams,
            )

        else:
            print("No config file used")

    def close(self):
        self.writer.writer.close()
