import torch
import torch.nn as nn
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

    @property
    @abc.abstractmethod
    def _objs_to_save(self):
        return {}

    @property
    def _objs_to_save_checked(self):
        save_objs = self._objs_to_save
        assert all(hasattr(self, key) for key in save_objs.keys())
        return save_objs

    def _add_to_save_dict(
            self,
            save_dict: dict,
            key: str,
            obj,
    ):
        assert key not in save_dict.keys()
        if isinstance(obj, MyObjectBase):
            save_dict[key] = obj.create_save_dict()
        else:
            save_dict[key] = obj

    def _set_obj_attr(
            self,
            key: str,
            obj,
    ):
        attr = getattr(self, key)
        if isinstance(attr, MyObjectBase):
            try:
                assert hasattr(self, key)
            except:
                raise ValueError
            attr.process_save_dict(obj)
        else:
            setattr(self, key, obj)

    def create_save_dict(self) -> dict:
        save_obj = {}
        for key, obj in self._objs_to_save_checked.items():
            self._add_to_save_dict(
                save_dict=save_obj,
                key=key,
                obj=obj
            )
        return save_obj

    def load(self, file_name, base_dir='.'):
        save_path = self._get_save_path(
            file_name=file_name,
            base_dir=base_dir
        )
        save_obj = torch.load(save_path)
        assert isinstance(save_obj, dict)
        assert all(key in save_obj.keys() for key in self._objs_to_save_checked.keys())
        self.process_save_dict(save_obj)

    def process_save_dict(self, save_obj):
        for key, obj in save_obj.items():
            self._set_obj_attr(
                obj=save_obj[key],
                key=key,
            )
