from typing import Type
import math

from diayn_cont.algo.cont_algo import DIAYNContAlgo
from diayn_cont.post_epoch_funcs.df_memory_eval import DfMemoryEvalDIAYNCont
from diayn_cont.post_epoch_funcs.df_env_eval import DfEnvEvaluationDIAYNCont
from diayn_cont.data_collector.seq_eval_collector import MdpPathCollectorWithReset
from diayn_cont.policy.skill_policy_with_skill_selector import MakeDeterministic

from latent_with_splitseqs.post_epoch_funcs.tb_logging import PostEpochTbLogger
from latent_with_splitseqs.post_epoch_funcs.net_param_histogram_logging \
    import NetParamHistogramLogger
from latent_with_splitseqs.algo.post_epoch_func_gtstamp_wrapper \
    import post_epoch_func_wrapper
from latent_with_splitseqs.post_epoch_funcs.algo_saving import ConfigSaver
from latent_with_splitseqs.post_epoch_funcs.plot_saved_skills_distribution \
    import ReplayBufferSkillDistributionPlotter
from latent_with_splitseqs.post_epoch_funcs.object_logging import ObjectSaver
from latent_with_splitseqs.algo.add_post_epoch_func \
    import add_post_epoch_funcs, add_post_epoch_func

from self_sup_combined.base.writer.diagnostics_writer import \
    DiagnosticsWriter, DiagnosticsWriterBase

from my_utils.dicts.get_config_item import get_config_item


def get_algo(
        algo_class: Type[DIAYNContAlgo],
        algo_kwargs: dict,
        df,
        diagnostic_writer,
        eval_policy: MakeDeterministic,
        post_epoch_eval_path_collector: MdpPathCollectorWithReset,
        config: dict,

) -> DIAYNContAlgo:
    tb_log_interval = math.ceil(config['log_interval']/4)
    net_log_interval = get_config_item(
        config=config,
        key="net_log_interval",
        default=config['log_interval']
    )

    df_memory_eval =DfMemoryEvalDIAYNCont(
        replay_buffer=algo_kwargs['replay_buffer'],
        df_to_evaluate=df,
        diagnostics_writer=diagnostic_writer,
        **config['df_evaluation_memory'],
    )
    df_memory_eval = post_epoch_func_wrapper('df evaluation on memory')(df_memory_eval)
    df_env_eval = DfEnvEvaluationDIAYNCont(
        seq_collector=post_epoch_eval_path_collector,
        seq_len=config['algorithm_kwargs']['max_path_length'],
        df_to_evaluate=df,
        diagnostic_writer=diagnostic_writer,
        log_prefix=None,
        **config['df_evaluation_env'],
    )
    df_env_eval = post_epoch_func_wrapper('df evaluation on env')(df_env_eval)

    net_logger = ObjectSaver(
        diagnostic_writer=diagnostic_writer,
        objects_periodic_saving=dict(
            policy_net=eval_policy,
        ),
        objects_initial_saving=dict(
            env=algo_kwargs['exploration_env'],
            config=config,
        )
    )
    net_logger = post_epoch_func_wrapper(
        'object saving',
        log_interval=net_log_interval,
    )(net_logger)

    net_param_hist_logger = NetParamHistogramLogger(
        diagnostic_writer=diagnostic_writer,
        trainer=algo_kwargs['trainer'],
    )
    net_param_hist_logger = post_epoch_func_wrapper(
        'net parameter histogram logging',
        log_interval=tb_log_interval
    )(net_param_hist_logger)

    post_epoch_tb_logger = PostEpochTbLogger(
        diagnostic_writer=diagnostic_writer,
        trainer=algo_kwargs['trainer'],
        replay_buffer=algo_kwargs['replay_buffer'],
    )
    post_epoch_tb_logger = post_epoch_func_wrapper(
        'tb logging',
        log_interval=tb_log_interval
    )(post_epoch_tb_logger)

    config_saver = ConfigSaver(
        diagnostic_writer=diagnostic_writer,
        config=config,
    )
    config_saver = post_epoch_func_wrapper('config saving')(config_saver)

    saved_skill_dist_plotter = ReplayBufferSkillDistributionPlotter(
        diagnostic_writer=diagnostic_writer,
        replay_buffer=algo_kwargs['replay_buffer'],
    )
    saved_skill_dist_plotter = post_epoch_func_wrapper(
        'replay buffer skill dist plotting',
        log_interval=config['log_interval'],
    )(saved_skill_dist_plotter)

    algo_class = add_post_epoch_funcs([
        df_env_eval,
        df_memory_eval,
        net_logger,
        net_param_hist_logger,
        post_epoch_tb_logger,
        config_saver,
        saved_skill_dist_plotter,
    ])(algo_class)

    algorithm = algo_class(
        **algo_kwargs,
    )

    return algorithm
