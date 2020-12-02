from rlkit.torch.sac.diayn.diayn import DIAYNTrainer

from latent_with_splitseqs.base.my_object_base import MyObjectBase


class DIAYNTrainerBaseObject(DIAYNTrainer, MyObjectBase):

    @property
    def _objs_to_save(self):
        objs_to_save = super()._objs_to_save
        return dict(
            **objs_to_save,
            log_alpha=self.log_alpha,
            alpha_optimizer=self.alpha_optimizer,
            use_automatic_entropy_tuning=self.use_automatic_entropy_tuning,
            **self.get_snapshot(),
        )
