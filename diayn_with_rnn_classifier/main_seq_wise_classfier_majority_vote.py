import argparse
import torch
import numpy as np
import copy
#from gym.envs.mujoco import HalfCheetahEnv

import rlkit.torch.pytorch_util as ptu
from rlkit.launchers.launcher_util import setup_logger

from self_supervised.utils.writer import MyWriter
from self_supervised.env_wrapper.rlkit_wrapper import NormalizedBoxEnvWrapper
from self_supervised.network.flatten_mlp import FlattenMlp as \
    MyFlattenMlp
from self_sup_combined.base.writer.diagnostics_writer import DiagnosticsWriter
from self_sup_comb_discrete_skills.memory.replay_buffer_discrete_skills import \
    SelfSupervisedEnvSequenceReplayBufferDiscreteSkills

from diayn_original_tb.seq_path_collector.rkit_seq_path_collector import SeqCollector
from diayn_original_tb.policies.diayn_policy_extension import \
    MakeDeterministicExtension
from diayn_original_tb.policies.self_sup_policy_wrapper import RlkitWrapperForMySkillPolicy

from diayn_with_rnn_classifier.reward_calculation.reward_calculator import RewardPolicyDiff
from diayn_with_rnn_classifier.networks.rnn_classifier import SeqEncoder
from diayn_with_rnn_classifier.trainer.diayn_trainer_with_rnn_classifier import \
    DIAYNTrainerRnnClassifierExtension
from diayn_with_rnn_classifier.algo.seq_wise_algo_classfier_perf_logging import \
    SeqWiseAlgoClassfierPerfLogging
from diayn_with_rnn_classifier.policies.action_log_prob_calculator import \
    ActionLogpropCalculator
from diayn_with_rnn_classifier.trainer.seq_wise_trainer_with_diayn_classifier_vote import \
    DIAYNTrainerMajorityVoteSeqClassifier


def experiment(variant, args):
    expl_env = NormalizedBoxEnvWrapper(gym_id=str(args.env))
    eval_env = copy.deepcopy(expl_env)
    obs_dim = expl_env.observation_space.low.size
    action_dim = eval_env.action_space.low.size
    skill_dim = args.skill_dim

    seq_len = 100
    variant['algorithm_kwargs']['batch_size'] //= seq_len

    run_comment = ""
    run_comment += "seq_len: {} |  ".format(seq_len)
    run_comment += "own functions | "
    run_comment += "majority vote | "

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
    df = MyFlattenMlp(
        input_size=obs_dim,
        output_size=skill_dim,
        hidden_sizes=[M, M],
    )
    #policy = SkillTanhGaussianPolicyExtension(
    #    obs_dim=obs_dim + skill_dim,
    #    action_dim=action_dim,
    #    hidden_sizes=[M, M],
    #    skill_dim=skill_dim
    #)
    policy = RlkitWrapperForMySkillPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        skill_dim=skill_dim,
        hidden_sizes=[M, M],
    )
    eval_policy = MakeDeterministicExtension(policy)
    eval_path_collector = SeqCollector(
        eval_env,
        eval_policy,
    )
    expl_step_collector = SeqCollector(
        expl_env,
        policy,
    )
    seq_eval_collector = SeqCollector(
        env=eval_env,
        policy=eval_policy
    )
    replay_buffer = SelfSupervisedEnvSequenceReplayBufferDiscreteSkills(
        max_replay_buffer_size=variant['replay_buffer_size'],
        seq_len=seq_len,
        mode_dim=skill_dim,
        env=expl_env,
    )
    trainer = DIAYNTrainerMajorityVoteSeqClassifier(
        env=eval_env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        df=df,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        **variant['trainer_kwargs']
    )

    writer = MyWriter(
        seed=seed,
        log_dir='logs',
        run_comment=run_comment
    )
    diagno_writer = DiagnosticsWriter(
        writer=writer,
        log_interval=1
    )

    algorithm = SeqWiseAlgoClassfierPerfLogging(
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--env',
                        type=str,
                        default="MountainCarContinuous-v0",
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
        replay_buffer_size=int(1E6),
        algorithm_kwargs=dict(
            num_epochs=1000,
            num_eval_steps_per_epoch=5000,
            num_trains_per_train_loop=10,
            num_expl_steps_per_train_loop=10,
            min_num_steps_before_training=1000,
            max_path_length=1000,
            batch_size=1024,
        ),
        trainer_kwargs=dict(
            discount=0.99,
            soft_target_tau=5e-3,
            target_update_period=1,
            policy_lr=3E-4,
            qf_lr=3E-4,
            reward_scale=1,
            use_automatic_entropy_tuning=True,
        ),
    )
    setup_logger('DIAYN_' + str(args.skill_dim) + '_' + args.env, variant=variant)
    ptu.set_gpu_mode(True)  # optionally set the GPU (default=False)
    experiment(variant, args)
