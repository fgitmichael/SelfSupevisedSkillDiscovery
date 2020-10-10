import torch
import numpy as np
import copy

import rlkit.torch.pytorch_util as ptu
from rlkit.launchers.launcher_util import setup_logger

from self_supervised.env_wrapper.rlkit_wrapper import NormalizedBoxEnvWrapper
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

from latent_with_splitseqs.algo.algo_latent_splitseqs_with_eval \
    import SeqwiseAlgoRevisedSplitSeqsEval
from latent_with_splitseqs.data_collector.seq_collector_split import SeqCollectorSplitSeq
from latent_with_splitseqs.networks.seqwise_splitseq_classifier_rnn_end_recon_only \
    import SeqwiseSplitseqClassifierRnnEndReconOnly
from latent_with_splitseqs.trainer.rnn_with_splitseqs_trainer_end_recon_only \
    import URLTrainerRnnWithSplitseqsEndReconOnly

from two_d_navigation_demo.env.navigation_env import TwoDimNavigationEnv

def experiment(variant,
               config,
               config_path_name,
               ):
    expl_env = NormalizedBoxEnvWrapper(gym_id=variant['env_id'])
    eval_env = copy.deepcopy(expl_env)
    obs_dim = expl_env.observation_space.low.size
    action_dim = eval_env.action_space.low.size
    feature_dim_or_obs_dim = config.hidden_size_rnn \
        if variant['trainer_kwargs']['train_sac_in_feature_space'] \
        else obs_dim

    seq_len = config.seq_len
    skill_dim = config.skill_dim
    used_obs_dims_policy = tuple(i for i in range(obs_dim))
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
    df = SeqwiseSplitseqClassifierRnnEndReconOnly(
        seq_len=seq_len,
        obs_dim=obs_dim,
        skill_dim=skill_dim,
        hidden_units_classifier=config.hidden_units_classifier,
        hidden_size_rnn=config.hidden_size_rnn,
        leaky_slope_classifier=config.leaky_slope_classifier,
        dropout=config.classifier_dropout,
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

    replay_buffer = LatentReplayBuffer(
        max_replay_buffer_size=variant['replay_buffer_size'],
        seq_len=seq_len,
        mode_dim=skill_dim,
        env=expl_env,
    )
    loss_fun = GuidedInfoLoss(
        alpha=config.info_loss.alpha,
        lamda=config.info_loss.lamda,
    ).loss
    trainer = URLTrainerLatentWithSplitseqsRnnEndReconOnly(
        #skill_prior_dist=skill_prior,
        env=eval_env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        df=df,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        loss_fun=loss_fun,
        skill_prior_dist=skill_prior,
        **variant['trainer_kwargs']
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

    algorithm = SeqwiseAlgoRevisedSplitSeqsEval(
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
        default="config/standard_gym_env_id/ant_rnn_sac_in_latent_space.yaml",
        return_config_path_name=True,
    )

    # noinspection PyTypeChecker
    variant = dict(
        env_id=config.env_id,
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
