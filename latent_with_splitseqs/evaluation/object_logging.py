from diayn_original_tb.algo.algo_diayn_tb import DIAYNTorchOnlineRLAlgorithmTb



def save_policy(log_interval=20):
    def save_policy(self: DIAYNTorchOnlineRLAlgorithmTb, epoch):
        self.diagnostic_writer.save_object(
            obj=self.policy,
            save_name="policy_net",
            epoch=epoch,
            log_interval=log_interval,
        )
    return save_policy

def save_env(self: DIAYNTorchOnlineRLAlgorithmTb, epoch):
    if epoch == 0:
        self.diagnostic_writer.save_env(self.expl_env)
