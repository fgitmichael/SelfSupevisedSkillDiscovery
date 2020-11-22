from latent_with_splitseqs.base.tb_logging_base import PostEpochDiagnoWritingBase


class NetLogger(PostEpochDiagnoWritingBase):

    def __init__(self,
                 net_dict: dict,
                 env,
                 **kwargs
                 ):
        super(NetLogger, self).__init__(**kwargs)
        self.net_dict = net_dict
        self.env = env

    def __call__(self, *args, epoch, **kwargs):
        for name, net in self.net_dict.items():
            self.diagno_writer.save_object(
                obj=net,
                save_name=name,
                epoch=epoch,
            )

        if epoch == 0:
            self.diagno_writer.save_env(self.env)
