import torch
import torch.nn as nn
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

from seqwise_cont_skillspace.data_collector.skill_selector_cont_skills import \
    SkillSelectorContinous
from seqwise_cont_skillspace.utils.info_loss import GuidedInfoLoss

from mode_disent_no_ssm.utils.parse_args import parse_args

from latent_with_splitseqs.data_collector.seq_collector_split import SeqCollectorSplitSeq
from latent_with_splitseqs.algo.algo_latent_splitseq_with_eval_on_used_obsdim \
    import SeqwiseAlgoRevisedSplitSeqsEvalOnUsedObsDim

from latent_with_splitseqs.config.fun.get_env import get_env
from latent_with_splitseqs.latent.srnn_latent_conditioned_on_skill_seq \
    import SRNNLatentConditionedOnSkillSeq
from latent_with_splitseqs.networks.seqwise_splitseq_classifier_srnn_whole_seq_recon \
    import SplitSeqClassifierSRNNWholeSeqRecon
from latent_with_splitseqs.trainer.latent_srnn_full_seq_recon_trainer \
    import URLTrainerLatentSplitSeqsSRNNFullSeqRecon
from latent_with_splitseqs.latent.slac_latent_conditioned_on_skill_seq \
    import SlacLatentNetConditionedOnSkillSeq


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

    latent1_dim = 8
    latent2_dim = 32
    hidden_size_rnn = 10

    seq_len = config.seq_len
    skill_dim = config.skill_dim
    feature_dim_or_obs_dim = latent1_dim + latent2_dim + hidden_size_rnn

    used_obs_dims_policy = tuple([i for i in range(obs_dim)])

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
    skill_prior = ConstantGaussianMultiDim(
        output_dim=skill_dim,
    )
    skill_selector = SkillSelectorContinous(
        prior_skill_dist=skill_prior,
        grid_radius_factor=1.,
    )
    eval_path_collector = SeqCollectorSplitSeq(
        eval_env,
        eval_policy,
        max_seqs=1000,
        skill_selector=skill_selector
    )
    expl_step_collector = SeqCollectorSplitSeq(
        expl_env,
        policy,
        max_seqs=1000,
        skill_selector=skill_selector
    )
    seq_eval_collector = SeqCollectorSplitSeq(
        env=eval_env,
        policy=eval_policy,
        max_seqs=1000,
        skill_selector=skill_selector
    )
    loss_fun = GuidedInfoLoss(
        alpha=config.info_loss.alpha,
        lamda=config.info_loss.lamda,
    ).loss
    trainer_init_kwargs = dict(
        #skill_prior_dist=skill_prior,
        env=eval_env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        loss_fun=loss_fun,
        skill_prior_dist=skill_prior,
        **variant['trainer_kwargs']
    )
    latent_model_class_stoch = SlacLatentNetConditionedOnSkillSeq
    latent_model_det = nn.GRU(
        input_size=obs_dim,
        hidden_size=hidden_size_rnn,
        batch_first=True,
        bidirectional=False,
    )
    filter_net_params = dict(
        hidden_sizes=(256, 256),
        leaky_relu=0.2,
    )
    stochastic_latent_net_class_params = dict(
        latent1_dim=8,
        latent2_dim=32,
        dropout=0.1,
        beta_anneal=dict(
            start=0.2,
            add=0.0003,
            end=1,
        )
    )

    srnn_model = SRNNLatentConditionedOnSkillSeq(
        obs_dim=obs_dim,
        skill_dim=skill_dim,
        filter_net_params=filter_net_params,
        deterministic_latent_net=latent_model_det,
        stochastic_latent_net_class=latent_model_class_stoch,
        stochastic_latent_net_class_params=stochastic_latent_net_class_params,
    )
    df = SplitSeqClassifierSRNNWholeSeqRecon(
        seq_len=seq_len,
        obs_dim=obs_dim,
        skill_dim=skill_dim,
        latent_net=srnn_model,
        dropout_classifier=0.5,
        std_classifier=None,
    )
    trainer = URLTrainerLatentSplitSeqsSRNNFullSeqRecon(
        df=df,
        **trainer_init_kwargs
    )

    replay_buffer = LatentReplayBuffer(
        max_replay_buffer_size=variant['replay_buffer_size'],
        seq_len=seq_len,
        mode_dim=skill_dim,
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

    algorithm = SeqwiseAlgoRevisedSplitSeqsEvalOnUsedObsDim(
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

        seq_eval_len=config.seq_eval_len,
        horizon_eval_len=config.horizon_eval_len,

        **variant['algorithm_kwargs']
    )
    algorithm.to(ptu.device)
    algorithm.train()


if __name__ == "__main__":
    config, config_path_name = parse_args(
        default="config/all_in_one_config/two_d_nav/"
                "config_latent_normal_first_two_dims_slac_srnn.yaml",
        return_config_path_name=True,
    )

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
