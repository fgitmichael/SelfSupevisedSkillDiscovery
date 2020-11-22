from diayn_original_tb.algo.algo_diayn_tb import DIAYNTorchOnlineRLAlgorithmTb


def save_algo(self: DIAYNTorchOnlineRLAlgorithmTb, epoch):
    self.diagnostic_writer.save_object(
        self,
        save_name="last_algo.pkl",
        epoch=None,
    )
