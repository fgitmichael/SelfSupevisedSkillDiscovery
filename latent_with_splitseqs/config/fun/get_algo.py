from typing import Type
import math

from diayn_original_tb.algo.algo_diayn_tb import DIAYNTorchOnlineRLAlgorithmTb

from latent_with_splitseqs.post_epoch_funcs.tb_logging import PostEpochTbLogger
from latent_with_splitseqs.post_epoch_funcs.net_logging import NetLogger
from latent_with_splitseqs.post_epoch_funcs.df_env_eval import DfEnvEvaluationSplitSeq
from latent_with_splitseqs.post_epoch_funcs.net_param_histogram_logging \
    import NetParamHistogramLogger
from latent_with_splitseqs.post_epoch_funcs.df_memory_eval import DfMemoryEvalSplitSeq
from latent_with_splitseqs.algo.add_post_epoch_func import add_post_epoch_funcs
from latent_with_splitseqs.algo.post_epoch_func_gtstamp_wrapper \
    import post_epoch_func_wrapper
from latent_with_splitseqs.post_epoch_funcs.algo_saving import ConfigSaver, save_algo


def get_algo_with_post_epoch_funcs(
        algo_class_in: Type[DIAYNTorchOnlineRLAlgorithmTb],
        replay_buffer,
        expl_step_collector,
        eval_path_collector,
        seq_eval_collector,
        diagno_writer,
        eval_policy,
        df,
        config,
        expl_env,
        eval_env,
        trainer,
) -> DIAYNTorchOnlineRLAlgorithmTb:
    df_memory_eval = DfMemoryEvalSplitSeq(
        replay_buffer=replay_buffer,
        df_to_evaluate=df,
        diagnostics_writer=diagno_writer,
        **config.df_evaluation_memory
    )
    df_env_eval = DfEnvEvaluationSplitSeq(
        seq_collector=seq_eval_collector,
        df_to_evaluate=df,
        diagnostic_writer=diagno_writer,
        log_prefix=None,
        **config.df_evaluation_env,
    )
    net_logger = NetLogger(
        diagnostic_writer=diagno_writer,
        net_dict=dict(
            policy_net=eval_policy,
        ),
        env=expl_env
    )
    net_param_hist_logger = NetParamHistogramLogger(
        diagnostic_writer=diagno_writer,
        trainer=trainer
    )
    post_epoch_tb_logger = PostEpochTbLogger(
        diagnostic_writer=diagno_writer,
        trainer=trainer,
        replay_buffer=replay_buffer,
    )
    config_saver = ConfigSaver(
        diagno_writer=diagno_writer,
        config=config,
    )
    tb_log_interval = math.ceil(config.log_interval/4)
    algo_class = add_post_epoch_funcs([
        post_epoch_func_wrapper
        ('df evaluation on env')(df_env_eval),
        post_epoch_func_wrapper
        ('df evaluation on memory')(df_memory_eval),
        post_epoch_func_wrapper
        ('object saving')(net_logger),
        post_epoch_func_wrapper
        ('net parameter histogram logging',
         log_interval=tb_log_interval)(net_param_hist_logger),
        post_epoch_func_wrapper
        ('tb logging', log_interval=tb_log_interval)(post_epoch_tb_logger),
        post_epoch_func_wrapper
        ('config saving')(config_saver),
        post_epoch_func_wrapper
        ('algo logging', log_interval=config.log_interval * 100, method=True)(save_algo),
    ])(algo_class_in)

    algorithm = algo_class(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_step_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,

        seq_len=config.seq_len,
        horizon_len=config.horizon_len,

        diagnostic_writer=diagno_writer,
        seq_eval_collector=seq_eval_collector,

        **config.algorithm_kwargs,
    )

    return algorithm
