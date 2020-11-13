import torch
import numpy as np
import copy

import rlkit.torch.pytorch_util as ptu
from rlkit.launchers.launcher_util import setup_logger

from self_supervised.utils.writer import MyWriterWithActivation
from self_supervised.network.flatten_mlp import FlattenMlp as \
    MyFlattenMlp
from self_sup_combined.base.writer.diagnostics_writer import DiagnosticsWriter

from latent_with_splitseqs.memory.replay_buffer_for_latent import LatentReplayBuffer

from diayn_seq_code_revised.policies.skill_policy import \
    MakeDeterministicRevised
from diayn_seq_code_revised.networks.my_gaussian import ConstantGaussianMultiDim
from diayn_seq_code_revised.policies.skill_policy_obsdim_select \
    import SkillTanhGaussianPolicyRevisedObsSelect

from seqwise_cont_skillspace.networks.contant_uniform import ConstantUniformMultiDim
from seqwise_cont_skillspace.data_collector.skill_selector_cont_skills import \
    SkillSelectorContinous
from seqwise_cont_skillspace.utils.info_loss import GuidedInfoLoss

from mode_disent_no_ssm.utils.parse_args import parse_args, parse_args_hptuning

from latent_with_splitseqs.data_collector.seq_collector_split import SeqCollectorSplitSeq
from latent_with_splitseqs.config.fun.get_env import get_env
from latent_with_splitseqs.config.fun.get_obs_dims_used_policy \
    import get_obs_dims_used_policy
from latent_with_splitseqs.config.fun.get_df_and_trainer import get_df_and_trainer
from latent_with_splitseqs.config.fun.get_feature_dim_obs_dim \
    import get_feature_dim_obs_dim
from latent_with_splitseqs.utils.loglikelihoodloss import GuidedKldLogOnlyLoss
from latent_with_splitseqs.evaluation.df_memory_eval import DfMemoryEvalSplitSeq
from latent_with_splitseqs.algo.add_post_epoch_func import add_post_epoch_funcs
from latent_with_splitseqs.evaluation.net_param_histogram_logging \
    import NetParamHistogramLogger
from latent_with_splitseqs.algo.algo_latent_splitseqs import \
    SeqwiseAlgoRevisedSplitSeqs
from latent_with_splitseqs.evaluation.df_env_eval import DfEnvEvaluationSplitSeq
from latent_with_splitseqs.algo.post_epoch_func_gtstamp_wrapper \
    import post_epoch_func_wrapper
from latent_with_splitseqs.evaluation.net_logging import NetLogger
from latent_with_splitseqs.config.fun.get_skill_prior import get_skill_prior
from latent_with_splitseqs.config.fun.get_loss_fun import get_loss_fun

def experiment(variant,
               config,
               config_path_name,
               ):
    expl_env = get_env(
        **config.env_kwargs
    )
    eval_env = copy.deepcopy(expl_env)
    obs_dim = expl_env.observation_space.low.size
    action_dim = eval_env.action_space.low.size

    feature_dim_or_obs_dim = get_feature_dim_obs_dim(
        obs_dim=obs_dim,
        config=config,
    )
    used_obs_dims_policy = get_obs_dims_used_policy(
        obs_dim=obs_dim,
        config=config,
    )
    variant['algorithm_kwargs']['batch_size'] //= config.seq_len

    test_script_path_name = config.test_script_path \
        if "test_script_path" in config.keys() \
        else None

    sep_str = " | "
    run_comment = sep_str
    run_comment += "seq_len: {}".format(config.seq_len) + sep_str
    run_comment += config.algorithm + sep_str
    run_comment += config.version + sep_str

    log_folder=config.log_folder
    seed = 0
    torch.manual_seed = seed
    expl_env.seed(seed)
    eval_env.seed(seed)
    np.random.seed(seed)

    M = variant['layer_size']
    qf1 = MyFlattenMlp(
        input_size=feature_dim_or_obs_dim + action_dim + config.skill_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    qf2 = MyFlattenMlp(
        input_size=feature_dim_or_obs_dim + action_dim + config.skill_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    target_qf1 = MyFlattenMlp(
        input_size=feature_dim_or_obs_dim + action_dim + config.skill_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    target_qf2 = MyFlattenMlp(
        input_size=feature_dim_or_obs_dim + action_dim + config.skill_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    policy = SkillTanhGaussianPolicyRevisedObsSelect(
        obs_dim=len(used_obs_dims_policy),
        action_dim=action_dim,
        skill_dim=config.skill_dim,
        hidden_sizes=[M, M],
        obs_dims_selected=used_obs_dims_policy,
        obs_dim_real=obs_dim,
    )
    eval_policy = MakeDeterministicRevised(policy)
    skill_prior_for_loss = ConstantGaussianMultiDim(
        output_dim=config.skill_dim,
    )
    skill_prior = get_skill_prior(config)
    skill_selector = SkillSelectorContinous(
        prior_skill_dist=skill_prior,
        grid_radius_factor=1.,
    )
    eval_path_collector = SeqCollectorSplitSeq(
        eval_env,
        eval_policy,
        max_seqs=5000,
        skill_selector=skill_selector
    )
    expl_step_collector = SeqCollectorSplitSeq(
        expl_env,
        policy,
        max_seqs=5000,
        skill_selector=skill_selector
    )
    seq_eval_collector = SeqCollectorSplitSeq(
        env=eval_env,
        policy=eval_policy,
        max_seqs=5000,
        skill_selector=skill_selector
    )
    loss_fun = get_loss_fun(config)
    trainer_init_kwargs = dict(
        env=eval_env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        loss_fun=loss_fun,
        skill_prior_dist=skill_prior_for_loss,
        **variant['trainer_kwargs']
    )
    df, trainer = get_df_and_trainer(
        obs_dim=obs_dim,
        trainer_init_kwargs=trainer_init_kwargs,
        **config
    )

    replay_buffer = LatentReplayBuffer(
        max_replay_buffer_size=variant['replay_buffer_size'],
        seq_len=config.seq_len,
        mode_dim=config.skill_dim,
        env=expl_env,
    )

    writer = MyWriterWithActivation(
        seed=seed,
        log_dir=log_folder,
        run_comment=run_comment
    )
    diagno_writer = DiagnosticsWriter(
        writer=writer,
        log_interval=config.log_interval,
        config=config,
        config_path_name=config_path_name,
        test_script_path_name=test_script_path_name,
    )

    df_memory_eval = DfMemoryEvalSplitSeq(
        replay_buffer=replay_buffer,
        df_to_evaluate=df,
        diagnostics_writer=diagno_writer,
        **config.df_evaluation_memory
    )
    df_env_eval = DfEnvEvaluationSplitSeq(
        seq_collector=seq_eval_collector,
        df_to_evaluate=df,
        diagnostics_writer=diagno_writer,
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
    algo_class = add_post_epoch_funcs([
        post_epoch_func_wrapper
        ('df evaluation on env')(df_env_eval),
        post_epoch_func_wrapper
        ('df evaluation on memory')(df_memory_eval),
        post_epoch_func_wrapper
        ('object saving')(net_logger),
        post_epoch_func_wrapper
        ('net parameter histogram logging')(net_param_hist_logger),
    ])(SeqwiseAlgoRevisedSplitSeqs)
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

        **variant['algorithm_kwargs']
    )
    algorithm.to(ptu.device)
    algorithm.train()

    diagno_writer.close()


if __name__ == "__main__":
    config, config_path_name = parse_args_hptuning(
        default="config/all_in_one_config/mountaincar/"
                "rnn_v2.yaml",
        default_min="./config/all_in_one_config/mountaincar/"
                    "random_hp_search/"
                    "srnn_v0_min.yaml",
        default_max="./config/all_in_one_config/mountaincar/"
                    "random_hp_search/"
                    "srnn_v0_max.yaml",
        default_hp_tuning=False,
        return_config_path_name=True,
    )

    if config.random_hp_tuning:
        #config.latent_kwargs.latent2_dim = config.latent_kwargs.latent1_dim * 8

        config.df_evaluation_env.seq_len = config.seq_len
        config.df_evaluation_memory.seq_len = config.seq_len
        #config.horizon_len = max(100,
        #                         np.random.randint(1, 20) * config.seq_len)
        config.df_evaluation_env.horizon_len = config.horizon_len
        config.df_evaluation_memory.horizon_len = config.horizon_len

        classifier_layer_size = np.random.randint(32, 256)
        config.df_kwargs_srnn.hidden_units_classifier = [classifier_layer_size,
                                                        classifier_layer_size]

        #if np.random.choice([True, False]):
        #    config.df_type.feature_extractor = 'rnn'
        #    config.df_type.latent_type = None

        #else:
        #    config.df_type.feature_extractor = 'latent_slac'
        #    config.df_type.rnn_type = None
        #    if np.random.choice([True, False]):
        #        config.df_type.latent_type = 'single_skill'
        #    else:
        #        config.df_type.latent_type = 'full_seq'

        #if np.random.choice([True, False]):
        #    config.algorithm_kwargs.train_sac_in_feature_space = False

        config_path_name = None

        #if np.random.choice([True, False]):
        #    config.df_kwargs_srnn.std_classifier = np.random.rand() + 0.3

    config.horizon_len = (config.horizon_len // config.seq_len) * config.seq_len
    config.df_evaluation_memory.horizon_len = config.horizon_len
    config.df_evaluation_env.horizon_len = config.horizon_len

    # noinspection PyTypeChecker
    variant = dict(
        algorithm=config.algorithm,
        version=config.version,
        layer_size=config.layer_size,
        replay_buffer_size=config.replay_buffer_size,
        algorithm_kwargs=config.algorithm_kwargs,
        trainer_kwargs=config.trainer_kwargs,
    )
    setup_logger(config.algorithm
                 + config.version
                 + str(config.skill_dim), variant=variant)
    ptu.set_gpu_mode(True)  # optionally set the GPU (default=False)

    experiment(variant,
               config,
               config_path_name)
