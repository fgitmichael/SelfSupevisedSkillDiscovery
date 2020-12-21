from latent_with_splitseqs.base.tb_logging_base import PostEpochDiagnoWritingBase


class ObjectSaver(PostEpochDiagnoWritingBase):

    def __init__(self,
                 objects_periodic_saving: dict,
                 objects_initial_saving,
                 **kwargs
                 ):
        super(ObjectSaver, self).__init__(**kwargs)
        self.periodic_saving_dict = objects_periodic_saving
        self.initial_saving_dict = objects_initial_saving

    def __call__(self, *args, epoch, **kwargs):
        if epoch == 0:
            for name, object in self.initial_saving_dict.items():
                self.diagno_writer.save_object(
                    obj=object,
                    save_name=name,
                    epoch=None,
                )

        for name, object in self.periodic_saving_dict.items():
            self.diagno_writer.save_object(
                obj=object,
                save_name=name,
                epoch=epoch,
            )
