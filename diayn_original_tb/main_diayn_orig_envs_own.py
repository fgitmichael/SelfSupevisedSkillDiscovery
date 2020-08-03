import gym
import argparse
import torch
import numpy as np
import copy
#from gym.envs.mujoco import HalfCheetahEnv

import rlkit.torch.pytorch_util as ptu
from rlkit.torch.sac.diayn.diayn_env_replay_buffer import DIAYNEnvReplayBuffer
from rlkit.launchers.launcher_util import setup_logger
from rlkit.torch.sac.diayn.diayn_path_collector import DIAYNMdpPathCollector
from rlkit.samplers.data_collector.step_collector import MdpStepCollector
from rlkit.torch.sac.diayn.diayn import DIAYNTrainer
from rlkit.torch.networks import FlattenMlp

from self_supervised.utils.writer import MyWriterWithActivation
from self_supervised.env_wrapper.rlkit_wrapper import NormalizedBoxEnvWrapper
from self_supervised.network.flatten_mlp import FlattenMlp as \
    MyFlattenMlp
from self_sup_combined.base.writer.diagnostics_writer import DiagnosticsWriter

from diayn_original_tb.seq_path_collector.rkit_seq_path_collector import SeqCollector
from diayn_original_tb.policies.diayn_policy_extension import \
    SkillTanhGaussianPolicyExtension, MakeDeterministicExtension
from diayn_original_tb.algo.algo_diayn_tb_perf_logging import \
    DIAYNTorchOnlineRLAlgorithmTbPerfLogging

from diayn_with_rnn_classifier.trainer.diayn_trainer_modularized import \
    DIAYNTrainerModularized


def experiment(variant, args):
    expl_env = NormalizedBoxEnvWrapper(gym_id=str(args.env))
    eval_env = copy.deepcopy(expl_env)
    obs_dim = expl_env.observation_space.low.size
    action_dim = eval_env.action_space.low.size
    skill_dim = args.skill_dim

    run_comment = ""
    run_comment += "DIAYN_policy | "
    run_comment += "DIAYN_mlp | "
    run_comment += "own_env | "
    run_comment += "perf_loggin | "

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
    qf2 = FlattenMlp(
        input_size=obs_dim + action_dim + skill_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    target_qf1 = FlattenMlp(
        input_size=obs_dim + action_dim + skill_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    target_qf2 = FlattenMlp(
        input_size=obs_dim + action_dim + skill_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    df = FlattenMlp(
        input_size=obs_dim,
        output_size=skill_dim,
        hidden_sizes=[M, M],
    )
    policy = SkillTanhGaussianPolicyExtension(
        obs_dim=obs_dim + skill_dim,
        action_dim=action_dim,
        hidden_sizes=[M, M],
        skill_dim=skill_dim
    )
    eval_policy = MakeDeterministicExtension(policy)
    eval_path_collector = DIAYNMdpPathCollector(
        eval_env,
        eval_policy,
    )
    expl_step_collector = MdpStepCollector(
        expl_env,
        policy,
    )
    seq_eval_collector = SeqCollector(
        env=eval_env,
        policy=eval_policy
    )
    replay_buffer = DIAYNEnvReplayBuffer(
        variant['replay_buffer_size'],
        expl_env,
        skill_dim
    )
    trainer = DIAYNTrainerModularized(
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
        log_dir='logs',
        run_comment=run_comment
    )
    diagno_writer = DiagnosticsWriter(
        writer=writer,
        log_interval=1
    )

    algorithm = DIAYNTorchOnlineRLAlgorithmTbPerfLogging(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_step_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,

        diagnostic_writer=diagno_writer,
        seq_eval_collector=seq_eval_collector,

        **variant['algorithm_kwargs']
    )
    algorithm.to(ptu.device)
    algorithm.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('env', type=str,
                        help='environment')
    parser.add_argument('--skill_dim', type=int, default=10,
                        help='skill dimension')
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
            num_trains_per_train_loop=1000,
            num_expl_steps_per_train_loop=1000,
            min_num_steps_before_training=1000,
            max_path_length=1000,
            batch_size=256,
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
