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

from diayn_original_cont.policy.policies import \
    SkillTanhGaussianPolicyExtensionCont, MakeDeterministicCont
from diayn_original_cont.trainer.diayn_cont_trainer import \
    DIAYNTrainerCont
from diayn_original_cont.algo.algo_cont import DIAYNContAlgo
from diayn_original_cont.trainer.info_loss_min_vae import \
    InfoLossLatentGuided
from diayn_original_cont.networks.vae_regressor import VaeRegressor
from diayn_original_cont.data_collector.seq_collector_optionally_id import \
    SeqCollectorRevisedOptionalId

from diayn_seq_code_revised.data_collector.skill_selector import \
    SkillSelectorDiscrete
from diayn_no_oh.utils.hardcoded_grid_two_dim import NoohGridCreator

from diayn_seq_code_revised.networks.my_gaussian import \
    ConstantGaussianMultiDim


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
    run_comment += "cont skills vae"
    run_comment += "skill_selector discrete | "
    run_comment += "info loss with guide"

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
    skill_prior = ConstantGaussianMultiDim(
        output_dim=skill_dim
    )
    noohgrid_creator_fun = NoohGridCreator().get_grid
    skill_selector = SkillSelectorDiscrete(
        get_skill_grid_fun=noohgrid_creator_fun
    )
    #skill_selector = SkillSelectorContinous(prior_skill_dist=skill_prior)
    policy = SkillTanhGaussianPolicyExtensionCont(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=[M, M],
        skill_dim=skill_dim,
        skill_selector_cont=skill_selector,
    )
    eval_policy = MakeDeterministicCont(policy)
    eval_path_collector = DIAYNMdpPathCollector(
        eval_env,
        eval_policy,
    )
    expl_step_collector = MdpStepCollector(
        expl_env,
        policy,
    )
    seq_eval_collector = SeqCollectorRevisedOptionalId(
        env=eval_env,
        policy=eval_policy,
        skill_selector=skill_selector,
        max_seqs=1000,
    )
    replay_buffer = DIAYNEnvReplayBuffer(
        variant['replay_buffer_size'],
        expl_env,
        skill_dim
    )
    info_loss_fun = InfoLossLatentGuided(
        alpha=0.99,
        lamda=0.2,
    ).loss
    df = VaeRegressor(
        input_size=obs_dim,
        latent_dim=skill_dim,
        output_size=obs_dim,
        hidden_sizes_enc=[30, 30],
    )
    trainer = DIAYNTrainerCont(
        env=eval_env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        df=df,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        info_loss_fun=info_loss_fun,
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

    algorithm = DIAYNContAlgo(
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
    parser.add_argument('--env',
                        type=str,
                        default="MountainCarContinuous-v0",
                        help='environment'
                        )
    parser.add_argument('--skill_dim',
                        type=int,
                        default=2,
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
