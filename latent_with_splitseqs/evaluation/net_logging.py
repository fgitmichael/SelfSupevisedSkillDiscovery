
class NetLogger(object):

    def __init__(self,
                 diagnostic_writer,
                 net_dict: dict,
                 env,
                 ):
        self.diagnostic_writer = diagnostic_writer
        self.net_dict = net_dict
        self.env = env

    def __call__(self, *args, epoch, **kwargs):
        for name, net in self.net_dict.items():
            self.diagnostic_writer.save_object(
                obj=net,
                save_name=name,
                epoch=epoch,
            )

        if epoch == 0:
            self.diagnostic_writer.save_env(self.env)
