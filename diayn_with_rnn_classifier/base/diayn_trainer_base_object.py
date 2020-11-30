from rlkit.torch.sac.diayn.diayn import DIAYNTrainer

from latent_with_splitseqs.base.my_object_base import MyObjectBase


class DIAYNTrainerBaseObject(DIAYNTrainer, MyObjectBase):

    @property
    def _objs_to_save(self):
        objs_to_save = super()._objs_to_save
        return dict(
            **objs_to_save,
            **self.get_snapshot()
        )
