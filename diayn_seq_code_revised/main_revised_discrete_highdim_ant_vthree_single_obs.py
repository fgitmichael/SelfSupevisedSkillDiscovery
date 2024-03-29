import argparse
import torch
import numpy as np
import copy
import os
import gym
from gym.envs.mujoco.ant_v3 import AntEnv as AntVersionThreeEnv
from my_utils.env_pixel_wrapper.mujoco_pixel_wrapper import MujocoPixelWrapper

import rlkit.torch.pytorch_util as ptu
from rlkit.launchers.launcher_util import setup_logger

from self_supervised.utils.writer import MyWriterWithActivation
from self_supervised.network.flatten_mlp import FlattenMlp as \
    MyFlattenMlp
from self_supervised.env_wrapper.rlkit_wrapper import NormalizedBoxEnvWrapper
from self_supervised.env_wrapper.pixel_wrapper import PixelNormalizedBoxEnvWrapper

from self_sup_combined.base.writer.diagnostics_writer import DiagnosticsWriter
from self_sup_comb_discrete_skills.memory.replay_buffer_discrete_skills import \
    SelfSupervisedEnvSequenceReplayBufferDiscreteSkills

#from diayn_rnn_seq_rnn_stepwise_classifier.networks.bi_rnn_stepwise_seqwise import \
#    BiRnnStepwiseSeqWiseClassifier
from diayn_seq_code_revised.networks.bi_rnn_stepwise_seqwise_obs_dimension_selection \
    import RnnStepwiseSeqwiseClassifierObsDimSelect

from diayn_seq_code_revised.data_collector.seq_collector_revised_discrete_skills import \
    SeqCollectorRevisedDiscreteSkills
from diayn_seq_code_revised.policies.skill_policy import \
    SkillTanhGaussianPolicyRevised, MakeDeterministicRevised
from diayn_seq_code_revised.data_collector.skill_selector import SkillSelectorDiscrete
from diayn_seq_code_revised.trainer.trainer_seqwise_stepwise_revised import \
    DIAYNAlgoStepwiseSeqwiseRevisedTrainer
from diayn_seq_code_revised.algo.seqwise_algo_revised_highdim import \
    SeqwiseAlgoRevisedDiscreteSkillsHighdim
from diayn_seq_code_revised.data_collector.seq_collector_revised_discreteskills_pixel \
    import SeqCollectorRevisedDiscreteSkillsPixel
from diayn_rnn_seq_rnn_stepwise_classifier.networks.bi_rnn_stepwise_seqwise import \
    BiRnnStepwiseSeqWiseClassifier
from diayn_seq_code_revised.trainer.trainer_seqwise_stepwise_revised_obsdim_single \
    import DIAYNAlgoStepwiseSeqwiseRevisedObsDimSingleTrainer

from diayn_seq_code_revised.policies.skill_policy_obsdim_select \
    import SkillTanhGaussianPolicyRevisedObsSelect
from diayn_seq_code_revised.networks.bi_rnn_stepwise_seqwise_singledims import \
    BiRnnStepwiseSeqWiseClassifierSingleDims

from diayn_no_oh.utils.hardcoded_grid_two_dim import NoohGridCreator, OhGridCreator


def experiment(variant, args):
    expl_env = AntVersionThreeEnv(
        exclude_current_positions_from_observation=False,
        xml_file = os.path.join(os.getcwd(), "./models/ant_reduced_gearratio.xml"),
    )
    eval_env = copy.deepcopy(expl_env)
    render_kwargs = dict(
        width=64,
        height=64,
    )
    pixel_env = MujocoPixelWrapper(
        env=copy.deepcopy(eval_env),
        render_kwargs=render_kwargs,
    )

    obs_dim = expl_env.observation_space.low.size
    action_dim = eval_env.action_space.low.size

    oh_grid_creator = OhGridCreator(
        num_skills=args.skill_dim,
    )
    get_oh_grid = oh_grid_creator.get_grid

    seq_len = 30
    skill_dim = args.skill_dim
    num_skills = args.skill_dim
    hidden_size_rnn = 30
    variant['algorithm_kwargs']['batch_size'] //= seq_len
    pos_encoding = "transformer"
    obs_dim_used_df = (0, 1,)
    obs_dim_used_policy = tuple(i for i in range(2, obs_dim))

    sep_str = " | "
    run_comment = sep_str
    run_comment += "seq_len: {}".format(seq_len) + sep_str
    run_comment += "seq wise step wise revised high dim" + sep_str
    run_comment += "hidden rnn_dim: {}{}".format(hidden_size_rnn, sep_str)
    run_comment += "pos encoding: {}{}".format(pos_encoding, sep_str)
    run_comment += "include current positions{}".format(sep_str)

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
    df = BiRnnStepwiseSeqWiseClassifierSingleDims(
        input_size=obs_dim,
        skill_dim=num_skills,
        feature_size=10,
        hidden_size_rnn=hidden_size_rnn,
        hidden_sizes_classifier_seq=[M, M],
        hidden_sizes_classifier_step=[M, M],
        hidden_size_feature_dim_matcher=[M,],
        seq_len=seq_len,
        pos_encoder_variant=pos_encoding,
        dropout=0.5,
        obs_dims_used=obs_dim_used_df,
    )
    policy = SkillTanhGaussianPolicyRevisedObsSelect(
        obs_dim=len(obs_dim_used_policy),
        action_dim=action_dim,
        skill_dim=skill_dim,
        hidden_sizes=[M, M],
        obs_dim_real=obs_dim,
        obs_dims_selected=obs_dim_used_policy,
    )
    eval_policy = MakeDeterministicRevised(policy)
    skill_selector = SkillSelectorDiscrete(
        get_skill_grid_fun=get_oh_grid
    )
    eval_path_collector = SeqCollectorRevisedDiscreteSkills(
        eval_env,
        eval_policy,
        max_seqs=50,
        skill_selector=skill_selector
    )
    expl_step_collector = SeqCollectorRevisedDiscreteSkills(
        expl_env,
        policy,
        max_seqs=50,
        skill_selector=skill_selector
    )
    seq_eval_collector = SeqCollectorRevisedDiscreteSkills(
        env=eval_env,
        policy=eval_policy,
        max_seqs=50,
        skill_selector=skill_selector
    )
    seqpixel_eval_collector = SeqCollectorRevisedDiscreteSkillsPixel(
        env=pixel_env,
        policy=eval_policy,
        max_seqs=50,
        skill_selector=skill_selector,
    )
    replay_buffer = SelfSupervisedEnvSequenceReplayBufferDiscreteSkills(
        max_replay_buffer_size=variant['replay_buffer_size'],
        seq_len=seq_len,
        mode_dim=skill_dim,
        env=expl_env,
    )
    trainer = DIAYNAlgoStepwiseSeqwiseRevisedObsDimSingleTrainer(
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
        log_dir='logshighdim_ant',
        run_comment=run_comment
    )
    diagno_writer = DiagnosticsWriter(
        writer=writer,
        log_interval=3
    )

    algorithm = SeqwiseAlgoRevisedDiscreteSkillsHighdim(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_step_collector,
        evaluation_data_collector=eval_path_collector,
        seqpixel_eval_collector=seqpixel_eval_collector,
        replay_buffer=replay_buffer,

        seq_len=seq_len,
        seq_len_eval=100,

        diagnostic_writer=diagno_writer,
        seq_eval_collector=seq_eval_collector,
        mode_influence_plotting=False,

        **variant['algorithm_kwargs']
    )
    algorithm.to(ptu.device)
    algorithm.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env',
                        type=str,
                        default="HalfCheetah-v2",
                        help='environment'
                        )
    parser.add_argument('--skill_dim',
                        type=int,
                        default=10,
                        help='skill dimension'
                        )
    args = parser.parse_args()

    # noinspection PyTypeChecker
    variant = dict(
        algorithm="DIAYN",
        version="normal",
        layer_size=256,
        replay_buffer_size=int(5000),
        algorithm_kwargs=dict(
            num_epochs=1000,
            num_eval_steps_per_epoch=5000,
            num_trains_per_train_loop=10,
            num_expl_steps_per_train_loop=10,
            min_num_steps_before_training=1000,
            max_path_length=1000,
            batch_size=1000,
        ),
        trainer_kwargs=dict(
            discount=0.99,
            soft_target_tau=5e-3,
            target_update_period=1,
            policy_lr=3E-4,
            qf_lr=3E-4,
            reward_scale=1,
            df_lr_seq=1E-3,
            df_lr_step=1E-3,
            use_automatic_entropy_tuning=True,
        ),
    )
    setup_logger('DIAYN_' + str(args.skill_dim) + '_' + args.env, variant=variant)
    ptu.set_gpu_mode(True)  # optionally set the GPU (default=False)
    experiment(variant, args)
