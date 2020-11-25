import torch
import abc
import os


class MyObjectBase(object, metaclass=abc.ABCMeta):

    def _prepare_file_name(self, file_name):
        file_name_root, file_name_ext = os.path.splitext(file_name)
        if file_name_ext == '':
            file_name_ext = '.pkl'
        file_name = file_name_root + file_name_ext
        return file_name

    def _get_save_path(self, base_dir, file_name):
        file_name = self._prepare_file_name(file_name)
        save_path = os.path.join(base_dir, file_name)
        return save_path

    def save(self, file_name, base_dir='.'):
        save_path = self._get_save_path(
            file_name=file_name,
            base_dir=base_dir
        )
        save_obj = self.create_save_dict()
        torch.save(save_obj, save_path)

    @abc.abstractmethod
    def create_save_dict(self) -> dict:
        return {}

    def load(self, file_name, base_dir='.'):
        save_path = self._get_save_path(
            file_name=file_name,
            base_dir=base_dir
        )
        save_obj = torch.load(save_path)
        self.process_save_dict(save_obj)

    @abc.abstractmethod
    def process_save_dict(self, save_obj):
        assert not hasattr(super(), '_load')
