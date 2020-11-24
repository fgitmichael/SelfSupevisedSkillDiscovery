import torch
import abc
import os


class MyObjectBase(object, metaclass=abc.ABCMeta):

    def save(self, file_name, base_dir='.'):
        # Check for file extension
        file_name_root, file_name_ext = os.path.splitext(file_name)
        if file_name_ext == '':
            file_name_ext = '.pkl'
        file_name = file_name_root + file_name_ext

        # Prepare save path
        save_path = os.path.join(base_dir, file_name)

        # Save
        save_obj = self._save()
        torch.save(save_obj, save_path)

    @abc.abstractmethod
    def _save(self) -> dict:
        return {}

    def load(self, file_name, base_dir='.'):
        save_path = os.path.join(base_dir, file_name)
        save_obj = torch.load(save_path)
        self._load(save_obj)

    @abc.abstractmethod
    def _load(self, save_obj):
        assert not hasattr(super(), '_load')
