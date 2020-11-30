import argparse
import torch
import numpy as np
import copy
from gym.envs.mujoco.swimmer_v3 import SwimmerEnv as SwimmerVersionThreeEnv
#from gym.envs.mujoco import HalfCheetahEnv

import rlkit.torch.pytorch_util as ptu
from rlkit.launchers.launcher_util import setup_logger

from self_supervised.utils.writer import MyWriterWithActivation
from self_supervised.network.flatten_mlp import FlattenMlp as \
    MyFlattenMlp
from self_sup_combined.base.writer.diagnostics_writer import DiagnosticsWriter

from self_supervised.memory.self_sup_replay_buffer import \
    SelfSupervisedEnvSequenceReplayBuffer
from self_supervised.env_wrapper.rlkit_wrapper import NormalizedBoxEnvWrapper

from diayn_seq_code_revised.policies.skill_policy import \
    MakeDeterministicRevised
from diayn_seq_code_revised.policies.skill_policy_obsdim_select \
    import SkillTanhGaussianPolicyRevisedObsSelect
from diayn_seq_code_revised.networks.my_gaussian import \
    ConstantGaussianMultiDim
from seqwise_cont_skillspace.algo.algo_cont_skillspace_highdim import \
    SeqwiseAlgoRevisedContSkillsHighDim

from seqwise_cont_skillspace.trainer.trainer_single_dims_cont import \
    ContSkillTrainerSeqwiseStepwiseSingleDims
from seqwise_cont_skillspace.networks.classifier_seq_step_cont_choose_dims import \
    RnnStepwiseSeqwiseClassifierObsDimSelect
from seqwise_cont_skillspace.utils.info_loss import InfoLoss, GuidedInfoLoss
from seqwise_cont_skillspace.data_collector.skill_selector_cont_skills import \
    SkillSelectorContinous
from seqwise_cont_skillspace.data_collector.seq_collector_optional_skill_id import \
    SeqCollectorRevisedOptionalSkillId
from seqwise_cont_skillspace.networks.contant_uniform import ConstantUniformMultiDim

from two_d_navigation_demo.env.navigation_env import TwoDimNavigationEnv

from mode_disent_no_ssm.utils.parse_args import parse_args

from seqwise_cont_skillspace.networks.bi_rnn_stepwise_seq_singledims_cont_output \
    import BiRnnStepwiseSeqWiseClassifierSingleDimsContOutput


def experiment(variant,
               config,
               config_path_name,
               ):
    expl_env = SwimmerVersionThreeEnv(
        exclude_current_positions_from_observation=False
    )
    eval_env = copy.deepcopy(expl_env)
    obs_dim = expl_env.observation_space.low.size
    action_dim = eval_env.action_space.low.size

    seq_len = config.seq_len
    skill_dim = config.skill_dim
    hidden_size_rnn = config.hidden_size_rnn
    used_obs_dims_df = config.obs_dims_used_df
    used_obs_dims_policy = tuple(i for i in range(2, obs_dim))
    variant['algorithm_kwargs']['batch_size'] //= seq_len

    test_script_path_name = config.test_script_path \
        if "test_script_path" in config.keys() \
        else None

    sep_str = " | "
    run_comment = sep_str
    run_comment += "seq_len: {}".format(seq_len) + sep_str
    run_comment += "continous skill space" + sep_str
    run_comment += "hidden rnn_dim: {}{}".format(hidden_size_rnn, sep_str)
    run_comment += "gused_obs_dimsuided latent loss"
    run_comment += "single dims"

    log_folder="logsswimmer"
    seed = 0
    torch.manual_seed = seed
    expl_env.seed(seed)
    eval_env.seed(seed)
    np.random.seed(seed)

    M = variant['layer_size']
    qf1 = MyFlattenMlp(
        input_size=obs_dim + action_dim + skill_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    qf2 = MyFlattenMlp(
        input_size=obs_dim + action_dim + skill_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    target_qf1 = MyFlattenMlp(
        input_size=obs_dim + action_dim + skill_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    target_qf2 = MyFlattenMlp(
        input_size=obs_dim + action_dim + skill_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    df = BiRnnStepwiseSeqWiseClassifierSingleDimsContOutput(
        input_size=obs_dim,
        skill_dim=skill_dim,
        hidden_size_rnn=hidden_size_rnn,
        feature_size=config.feature_size,
        hidden_sizes_classifier_seq=config.hidden_sizes_classifier_seq,
        hidden_sizes_classifier_step=config.hidden_sizes_classifier_step,
        hidden_size_feature_dim_matcher=config.hidden_size_feature_dim_matcher,
        seq_len=seq_len,
        pos_encoder_variant=config.pos_encoder_variant,
        dropout=config.dropout,
        obs_dims_used=used_obs_dims_df,
        bidirectional=config.rnn_bidirectional,
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
    eval_path_collector = SeqCollectorRevisedOptionalSkillId(
        eval_env,
        eval_policy,
        max_seqs=50,
        skill_selector=skill_selector
    )
    expl_step_collector = SeqCollectorRevisedOptionalSkillId(
        expl_env,
        policy,
        max_seqs=50,
        skill_selector=skill_selector
    )
    seq_eval_collector = SeqCollectorRevisedOptionalSkillId(
        env=eval_env,
        policy=eval_policy,
        max_seqs=50,
        skill_selector=skill_selector
    )
    replay_buffer = SelfSupervisedEnvSequenceReplayBuffer(
        max_replay_buffer_size=variant['replay_buffer_size'],
        seq_len=seq_len,
        mode_dim=skill_dim,
        env=expl_env,
    )
    info_loss_fun = GuidedInfoLoss(
        **config.info_loss
    ).loss
    trainer = ContSkillTrainerSeqwiseStepwiseSingleDims(
        skill_prior_dist=skill_prior,
        loss_fun=info_loss_fun,
        env=eval_env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        df=df,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        **variant['trainer_kwargs']
    )

    writer = MyWriterWithActivation(
        seed=seed,
        log_dir=log_folder,
        run_comment=run_comment
    )
    diagno_writer = DiagnosticsWriter(
        writer=writer,
        log_interval=30,
        config=config,
        config_path_name=config_path_name,
        scripts_to_copy=test_script_path_name,
    )

    algorithm = SeqwiseAlgoRevisedContSkillsHighDim(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_step_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,

        seq_len=seq_len,

        diagnostic_writer=diagno_writer,
        seq_eval_collector=seq_eval_collector,

        **variant['algorithm_kwargs']
    )
    algorithm.to(ptu.device)
    algorithm.train()


if __name__ == "__main__":
    config, config_path_name = parse_args(
        default="config/swimmer/guided_params_v1.yaml",
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
    setup_logger('Cont skill space guided' + str(config.skill_dim), variant=variant)
    ptu.set_gpu_mode(True)  # optionally set the GPU (default=False)
    experiment(
        variant,
        config,
        config_path_name,
    )