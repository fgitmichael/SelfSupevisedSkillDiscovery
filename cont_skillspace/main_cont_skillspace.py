import argparse
import torch
import numpy as np
import copy
#from gym.envs.mujoco import HalfCheetahEnv

import rlkit.torch.pytorch_util as ptu
from rlkit.launchers.launcher_util import setup_logger

from self_supervised.utils.writer import MyWriterWithActivation
from self_supervised.env_wrapper.rlkit_wrapper import NormalizedBoxEnvWrapper
from self_supervised.network.flatten_mlp import FlattenMlp as \
    MyFlattenMlp
from self_sup_combined.base.writer.diagnostics_writer import DiagnosticsWriter

from self_supervised.memory.self_sup_replay_buffer import \
    SelfSupervisedEnvSequenceReplayBuffer

from diayn_rnn_seq_rnn_stepwise_classifier.networks.bi_rnn_stepwise_seqwise import \
    BiRnnStepwiseSeqWiseClassifier

from diayn_seq_code_revised.policies.skill_policy import \
    SkillTanhGaussianPolicyRevised, MakeDeterministicRevised
from diayn_seq_code_revised.networks.my_gaussian import \
    ConstantGaussianMultiDim
from cont_skillspace.algo.algo_cont_skillspace import SeqwiseAlgoRevisedContSkills

from cont_skillspace.trainer.cont_skill_trainer import ContSkillTrainer
from cont_skillspace.networks.rnn_vae_classifier import \
    RnnVaeClassifierContSkills
from cont_skillspace.utils.info_loss import InfoLoss
from cont_skillspace.data_collector.skill_selector_cont_skills import \
    SkillSelectorContinous
from cont_skillspace.data_collector.seq_collector_optional_skill_id import \
    SeqCollectorRevisedOptionalSkillId


def experiment(variant, args):
    expl_env = NormalizedBoxEnvWrapper(gym_id=str(args.env))
    eval_env = copy.deepcopy(expl_env)
    obs_dim = expl_env.observation_space.low.size
    action_dim = eval_env.action_space.low.size


    seq_len = 100
    one_hot_skill_encoding = True
    #skill_dim = args.skill_dim
    skill_dim = 2
    hidden_size_rnn = 100
    variant['algorithm_kwargs']['batch_size'] //= seq_len

    sep_str = " | "
    run_comment = sep_str
    run_comment += "one hot: {}".format(one_hot_skill_encoding) + sep_str
    run_comment += "seq_len: {}".format(seq_len) + sep_str
    run_comment += "continous skill space" + sep_str
    run_comment += "hidden rnn_dim: {}{}".format(hidden_size_rnn, sep_str)

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
    df = RnnVaeClassifierContSkills(
        input_size=obs_dim,
        hidden_size_rnn=hidden_size_rnn,
        output_size=skill_dim,
        hidden_sizes=[M, M],
        feature_decode_hidden_size=[M, M],
        seq_len=seq_len
    )
    policy = SkillTanhGaussianPolicyRevised(
        obs_dim=obs_dim,
        action_dim=action_dim,
        skill_dim=skill_dim,
        hidden_sizes=[M, M],
    )
    eval_policy = MakeDeterministicRevised(policy)
    skill_prior = ConstantGaussianMultiDim(
        output_dim=skill_dim
    )
    skill_selector = SkillSelectorContinous(
        prior_skill_dist=skill_prior
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
        max_seqs = 50,
        skill_selector = skill_selector
    )
    replay_buffer = SelfSupervisedEnvSequenceReplayBuffer(
        max_replay_buffer_size=variant['replay_buffer_size'],
        seq_len=seq_len,
        mode_dim=skill_dim,
        env=expl_env,
    )
    info_loss_fun = InfoLoss(
        alpha=0.999,
        lamda=0.22
    ).loss
    trainer = ContSkillTrainer(
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
        log_dir='logs',
        run_comment=run_comment
    )
    diagno_writer = DiagnosticsWriter(
        writer=writer,
        log_interval=1
    )

    algorithm = SeqwiseAlgoRevisedContSkills(
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
