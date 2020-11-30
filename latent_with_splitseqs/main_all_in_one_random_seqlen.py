import torch
import torch.distributions as torch_dist
import numpy as np
import copy

import rlkit.torch.pytorch_util as ptu
from rlkit.launchers.launcher_util import setup_logger

from self_supervised.utils.writer import MyWriterWithActivation
from self_supervised.network.flatten_mlp import FlattenMlp as \
    MyFlattenMlp
from self_sup_combined.base.writer.diagnostics_writer import DiagnosticsWriter

from diayn_seq_code_revised.policies.skill_policy import \
    MakeDeterministicRevised
from diayn_seq_code_revised.networks.my_gaussian import ConstantGaussianMultiDim
from diayn_seq_code_revised.policies.skill_policy_obsdim_select \
    import SkillTanhGaussianPolicyRevisedObsSelect

from seqwise_cont_skillspace.data_collector.seq_collector_optional_skill_id import \
    SeqCollectorRevisedOptionalSkillId
from seqwise_cont_skillspace.networks.contant_uniform import ConstantUniformMultiDim

from seqwise_cont_skillspace.data_collector.skill_selector_cont_skills import \
    SkillSelectorContinous
from seqwise_cont_skillspace.utils.info_loss import GuidedInfoLoss

from mode_disent_no_ssm.utils.parse_args import parse_args, parse_args_hptuning

from latent_with_splitseqs.data_collector.seq_collector_split import SeqCollectorSplitSeq
from latent_with_splitseqs.algo.\
    algo_latent_splitseq_eval_on_used_obs_dim_random_evalseqlen import \
    SeqwiseAlgoRevisedSplitSeqsEvalOnUsedObsDimRandomSeqevalLen
from latent_with_splitseqs.config.fun.get_env import get_env
from latent_with_splitseqs.config.fun.get_obs_dims_used_policy \
    import get_obs_dims_used_policy
from latent_with_splitseqs.config.fun.get_df_and_trainer import get_df_and_trainer
from latent_with_splitseqs.config.fun.get_feature_dim_obs_dim \
    import get_feature_dim_obs_dim
from latent_with_splitseqs.memory.replay_buffer_latent_splitseq_sampling \
    import LatentReplayBufferSplitSeqSampling
from latent_with_splitseqs.utils.loglikelihoodloss import GuidedKldLogOnlyLoss


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

    seq_len = config.horizon_len
    skill_dim = config.skill_dim
    feature_dim_or_obs_dim = get_feature_dim_obs_dim(
        obs_dim=obs_dim,
        config=config,
    )
    used_obs_dims_policy = get_obs_dims_used_policy(
        obs_dim=obs_dim,
        config=config,
    )
    variant['algorithm_kwargs']['batch_size'] //= seq_len

    test_script_path_name = config.test_script_path \
        if "test_script_path" in config.keys() \
        else None

    sep_str = " | "
    run_comment = sep_str
    run_comment += "seq_len: {}".format(seq_len) + sep_str
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
        input_size=feature_dim_or_obs_dim + action_dim + skill_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    qf2 = MyFlattenMlp(
        input_size=feature_dim_or_obs_dim + action_dim + skill_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    target_qf1 = MyFlattenMlp(
        input_size=feature_dim_or_obs_dim + action_dim + skill_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    target_qf2 = MyFlattenMlp(
        input_size=feature_dim_or_obs_dim + action_dim + skill_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    policy = SkillTanhGaussianPolicyRevisedObsSelect(
        obs_dim=len(used_obs_dims_policy),
        action_dim=action_dim,
        skill_dim=skill_dim,
        hidden_sizes=[M, M],
        obs_dims_selected=used_obs_dims_policy,
        obs_dim_real=obs_dim,
    )
    eval_policy = MakeDeterministicRevised(policy)
    skill_prior_for_loss = ConstantGaussianMultiDim(
        output_dim=skill_dim,
    )
    skill_prior = ConstantUniformMultiDim(
        output_dim=skill_dim,
    )
    skill_selector = SkillSelectorContinous(
        prior_skill_dist=skill_prior,
        grid_radius_factor=1.,
    )
    eval_path_collector = SeqCollectorRevisedOptionalSkillId(
        eval_env,
        eval_policy,
        max_seqs=50000,
        skill_selector=skill_selector
    )
    expl_step_collector = SeqCollectorRevisedOptionalSkillId(
        expl_env,
        policy,
        max_seqs=50000,
        skill_selector=skill_selector
    )
    seq_eval_collector = SeqCollectorSplitSeq(
        env=eval_env,
        policy=eval_policy,
        max_seqs=50000,
        skill_selector=skill_selector
    )
    loss_fun = GuidedKldLogOnlyLoss(
        alpha=config.info_loss.alpha,
        lamda=config.info_loss.lamda,
    ).loss
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
        seq_len=seq_len,
        skill_dim=skill_dim,
        rnn_kwargs=config.rnn_kwargs,
        df_kwargs_rnn=config.df_kwargs_rnn,
        latent_kwargs=config.latent_kwargs,
        latent_kwargs_smoothing=config.latent_kwargs_smoothing,
        df_kwargs_latent=config.df_kwargs_latent,
        df_type=config.df_type,
        trainer_init_kwargs=trainer_init_kwargs,
        latent_single_layer_kwargs=config.latent_single_layer_kwargs,
    )

    replay_buffer = LatentReplayBufferSplitSeqSampling(
        max_replay_buffer_size=variant['replay_buffer_size'],
        seq_len=seq_len,
        mode_dim=skill_dim,
        env=expl_env,
        min_sample_seq_len=config.min_sample_seq_len,
        max_sample_seq_len=config.max_sample_seq_len,
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
        scripts_to_copy=test_script_path_name,
    )

    algorithm = SeqwiseAlgoRevisedSplitSeqsEvalOnUsedObsDimRandomSeqevalLen(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_step_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,

        seq_len=seq_len,
        horizon_len=config.horizon_len,

        diagnostic_writer=diagno_writer,
        seq_eval_collector=seq_eval_collector,

        seq_eval_len=None,
        horizon_eval_len=config.horizon_len,

        min_seq_eval_len=config.min_sample_seq_len,
        max_seq_eval_len=config.max_sample_seq_len,

        **variant['algorithm_kwargs']
    )
    algorithm.to(ptu.device)
    algorithm.train()


if __name__ == "__main__":
    config, config_path_name = parse_args_hptuning(
        default="config/all_in_one_config/two_d_nav/"
                "random_seq_len/config_latent_random_eval_len.yaml",
        default_max="config/all_in_one_config/swimmer/config_hp_search_max.yaml",
        default_min="config/all_in_one_config/swimmer/config_hpp_search_min.yaml",
        default_hp_tuning=False,
        return_config_path_name=True,
    )

    if config.random_hp_tuning:
        config.latent_kwargs.latent2_dim = config.latent_kwargs.latent1_dim * 8

        config.seq_eval_len = config.seq_len
        config.horizon_len = np.random.randint(1, 15) * config.seq_len
        config.horizon_eval_len = config.horizon_len

        if np.random.choice([True, False]):
            config.df_type.feature_extractor = 'rnn'
            config.df_type.latent_type = None

        else:
            config.df_type.feature_extractor = 'latent_slac'
            config.df_type.rnn_type = None
            if np.random.choice([True, False]):
                config.df_type.latent_type = 'single_skill'
            else:
                config.df_type.latent_type = 'full_seq'

        if np.random.choice([True, False]):
            config.algorithm_kwargs.train_sac_in_feature_space = False

        config_path_name = None

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
