import torch


from diayn_original_tb.algo.algo_diayn_tb import DIAYNTorchOnlineRLAlgorithmTb

if __name__ == "__main__":
    algo_file_name = "last_algo.pkl"
    algo_file = torch.load(algo_file_name, map_location=True)
    assert isinstance(algo_file, DIAYNTorchOnlineRLAlgorithmTb)

    # Continue
    algo_file.train(start_epoch=algo_file.epoch_cnt)
