import argparse
import torch
import numpy as np
import copy
#from gym.envs.mujoco import HalfCheetahEnv

import rlkit.torch.pytorch_util as ptu
from rlkit.launchers.launcher_util import setup_logger

from self_supervised.utils.writer import MyWriterWithActivation
from self_supervised.network.flatten_mlp import FlattenMlp as \
    MyFlattenMlp
from self_sup_combined.base.writer.diagnostics_writer import DiagnosticsWriter
from self_sup_comb_discrete_skills.memory.replay_buffer_discrete_skills import \
    SelfSupervisedEnvSequenceReplayBufferDiscreteSkills

from diayn_seq_code_revised.data_collector.seq_collector_revised_discrete_skills import \
    SeqCollectorRevisedDiscreteSkills
from diayn_seq_code_revised.policies.skill_policy import \
    SkillTanhGaussianPolicyRevised, MakeDeterministicRevised
from diayn_seq_code_revised.data_collector.skill_selector import SkillSelectorDiscrete

from diayn_no_oh.utils.hardcoded_grid_two_dim import OhGridCreator

from two_d_navigation_demo.env.navigation_env import \
    TwoDimNavigationEnv
from two_d_navigation_demo.algo.seqwise_algo_step_only import \
    AlgoStepwiseOnlyDiscreteSkills
from two_d_navigation_demo.trainer.trainer_stepwise_only_discrete import \
    StepwiseOnlyDiscreteTrainer

from cnn_classifier_stepwise.networks.classifier_cnn_feature_extractor_df import \
    CnnStepwiseClassifierDiscreteDf
from cnn_classifier_stepwise.networks.cnn_one_layer_classifier import \
    CnnFeatureExtractorTwoDim


def experiment(variant, args):
    expl_env = TwoDimNavigationEnv()
    eval_env = copy.deepcopy(expl_env)
    obs_dim = expl_env.observation_space.low.size
    action_dim = eval_env.action_space.low.size

    # Skill Grids

    oh_grid_creator = OhGridCreator(
        num_skills=args.skill_dim
    )
    get_oh_grid = oh_grid_creator.get_grid

    seq_len = 100
    one_hot_skill_encoding = False
    skill_dim = args.skill_dim
    num_skills = args.skill_dim
    variant['algorithm_kwargs']['batch_size'] //= seq_len

    sep_str = " | "
    run_comment = sep_str
    run_comment += "stepwise only {}".format(sep_str)
    run_comment += "one hot: {}".format(one_hot_skill_encoding) + sep_str
    run_comment += "seq_len: {}".format(seq_len) + sep_str
    run_comment += "seq wise step wise revised" + sep_str

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
    cnn_one_layer = CnnFeatureExtractorTwoDim(
        obs_dim=obs_dim,
        cnn_params=dict(
            channels=(10,),
            dropout=0.3,
        )
    )
    df = CnnStepwiseClassifierDiscreteDf(
        skill_dim=num_skills,
        hidden_sizes_classifier_step=[M, M],
        seq_len=seq_len,
        feature_extractor=cnn_one_layer,
        pos_encoder_variant='transformer',
        dropout=0.1,
    )
    policy = SkillTanhGaussianPolicyRevised(
        obs_dim=obs_dim,
        action_dim=action_dim,
        skill_dim=skill_dim,
        hidden_sizes=[M, M],
    )
    eval_policy = MakeDeterministicRevised(policy)
    skill_selector = SkillSelectorDiscrete(
        get_skill_grid_fun=get_oh_grid
    )
    eval_path_collector = SeqCollectorRevisedDiscreteSkills(
        eval_env, eval_policy,
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
        max_seqs = 50,
        skill_selector = skill_selector
    )
    replay_buffer = SelfSupervisedEnvSequenceReplayBufferDiscreteSkills(
        max_replay_buffer_size=variant['replay_buffer_size'],
        seq_len=seq_len,
        mode_dim=skill_dim,
        env=expl_env,
    )
    trainer = StepwiseOnlyDiscreteTrainer(
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
        log_dir='logs_stepwise_only',
        run_comment=run_comment
    )
    diagno_writer = DiagnosticsWriter(
        writer=writer,
        log_interval=1
    )

    algorithm = AlgoStepwiseOnlyDiscreteSkills(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_step_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,

        seq_len=seq_len,

        diagnostic_writer=diagno_writer,
        seq_eval_collector=seq_eval_collector,

        mode_influence_one_plot_scatter=True,

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
            df_lr_step=1E-3,
            reward_scale=1,
            use_automatic_entropy_tuning=True,
        ),
    )
    setup_logger('DIAYN_' + str(args.skill_dim) + '_' + args.env, variant=variant)
    ptu.set_gpu_mode(True)  # optionally set the GPU (default=False)
    experiment(variant, args)
