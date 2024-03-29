# PYTHONUNBUFFERED=1;LD_LIBRARY_PATH=$LD_LIBRARY_PATH:
# /home/michael/.mujoco/mujoco200/bin;LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so
import torch
import numpy as np
import copy

import rlkit.torch.pytorch_util as ptu
from rlkit.launchers.launcher_util import setup_logger

from self_supervised.network.flatten_mlp import FlattenMlp as \
    MyFlattenMlp

from diayn_seq_code_revised.policies.skill_policy import \
    MakeDeterministicRevised
from diayn_seq_code_revised.networks.my_gaussian import ConstantGaussianMultiDim
from diayn_seq_code_revised.policies.skill_policy_obsdim_select \
    import SkillTanhGaussianPolicyRevisedObsSelect

from seqwise_cont_skillspace.data_collector.skill_selector_cont_skills import \
    SkillSelectorContinous

from mode_disent_no_ssm.utils.parse_args import parse_args

from latent_with_splitseqs.data_collector.seq_collector_split import SeqCollectorSplitSeq
from latent_with_splitseqs.config.fun.get_env import get_env
from latent_with_splitseqs.config.fun.get_obs_dims_used_policy \
    import get_obs_dims_used_policy
from latent_with_splitseqs.config.fun.get_df_and_trainer import get_df_and_trainer
from latent_with_splitseqs.config.fun.get_feature_dim_obs_dim \
    import get_feature_dim_obs_dim
from latent_with_splitseqs.config.fun.get_skill_prior import get_skill_prior
from latent_with_splitseqs.config.fun.get_loss_fun import get_loss_fun
from latent_with_splitseqs.algo.algo_latent_split_horizon_expl_collection \
    import SeqwiseAlgoSplitHorizonExplCollection
from latent_with_splitseqs.config.fun.get_algo import get_algo_with_post_epoch_funcs
from latent_with_splitseqs.config.fun.get_replay_buffer_and_expl_collector \
    import get_replay_buffer_and_expl_collector
from latent_with_splitseqs.config.fun.get_diagnostics_writer import get_diagnostics_writer
from latent_with_splitseqs.config.fun.get_random_hp_params import get_random_hp_params
from latent_with_splitseqs.config.fun.prepare_hparams import prepare_hparams

from diayn_original_tb.algo.algo_diayn_tb import DIAYNTorchOnlineRLAlgorithmTb

from my_utils.dicts.get_config_item import get_config_item


def create_experiment(config,
                      config_path_name,
                      ) -> DIAYNTorchOnlineRLAlgorithmTb:
    expl_env = get_env(
        **config.env_kwargs
    )
    eval_env = copy.deepcopy(expl_env)
    obs_dim = expl_env.observation_space.low.size
    action_dim = eval_env.action_space.low.size

    feature_dim_or_obs_dim = get_feature_dim_obs_dim(
        obs_dim=obs_dim,
        config=config,
    )
    used_obs_dims_policy = get_obs_dims_used_policy(
        obs_dim=obs_dim,
        config=config,
    )

    test_script_path_name = config.test_script_path \
        if "test_script_path" in config.keys() \
        else None
    scripts_to_copy = config.scripts_to_copy \
        if "scripts_to_copy" in config.keys() \
        else None

    if test_script_path_name is not None and scripts_to_copy is not None:
        if isinstance(test_script_path_name, list):
            assert isinstance(scripts_to_copy, list)
            scripts_to_copy.extend(test_script_path_name)
        else:
            scripts_to_copy.append(test_script_path_name)

    elif test_script_path_name is not None and scripts_to_copy is None:
        scripts_to_copy = test_script_path_name

    replay_seq_sampling_variant = get_config_item(
        config=config,
        key='replay_seq_sampling',
    )

    sep_str = " | "
    run_comment = sep_str
    if not replay_seq_sampling_variant == 'sampling_random_seq_len':
        run_comment += "seq_len: {}".format(config.seq_len) + sep_str
    run_comment += config.algorithm + sep_str
    run_comment += config.version + sep_str

    seed = 0
    torch.manual_seed = seed
    expl_env.seed(seed)
    eval_env.seed(seed)
    np.random.seed(seed)

    M = config.layer_size
    qf1 = MyFlattenMlp(
        input_size=feature_dim_or_obs_dim + action_dim + config.skill_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    qf2 = MyFlattenMlp(
        input_size=feature_dim_or_obs_dim + action_dim + config.skill_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    target_qf1 = MyFlattenMlp(
        input_size=feature_dim_or_obs_dim + action_dim + config.skill_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    target_qf2 = MyFlattenMlp(
        input_size=feature_dim_or_obs_dim + action_dim + config.skill_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    policy = SkillTanhGaussianPolicyRevisedObsSelect(
        obs_dim=len(used_obs_dims_policy),
        action_dim=action_dim,
        skill_dim=config.skill_dim,
        hidden_sizes=[M, M],
        obs_dims_selected=used_obs_dims_policy,
        obs_dim_real=obs_dim,
    )
    eval_policy = MakeDeterministicRevised(policy)
    skill_prior_for_loss = ConstantGaussianMultiDim(
        output_dim=config.skill_dim,
    )
    skill_prior = get_skill_prior(config)
    skill_selector = SkillSelectorContinous(
        prior_skill_dist=skill_prior,
        grid_radius_factor=config.skill_prior.grid_radius_factor,
    )
    eval_path_collector = SeqCollectorSplitSeq(
        eval_env,
        eval_policy,
        max_seqs=5000,
        skill_selector=skill_selector,
    )
    seq_eval_collector = SeqCollectorSplitSeq(
        env=eval_env,
        policy=eval_policy,
        max_seqs=5000,
        skill_selector=skill_selector,
    )
    loss_fun = get_loss_fun(config)
    trainer_init_kwargs = dict(
        env=eval_env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        loss_fun=loss_fun,
        skill_prior_dist=skill_prior_for_loss,
        **config.trainer_kwargs,
    )
    df, trainer = get_df_and_trainer(
        obs_dim=obs_dim,
        trainer_init_kwargs=trainer_init_kwargs,
        **config
    )
    replay_buffer, expl_step_collector = get_replay_buffer_and_expl_collector(
        config=config,
        expl_env=expl_env,
        policy=policy,
        skill_selector=skill_selector,
    )
    diagno_writer = get_diagnostics_writer(
        run_comment=run_comment,
        config=config,
        scripts_to_copy=scripts_to_copy,
        seed=seed,
        config_path_name=config_path_name,
    )
    algorithm = get_algo_with_post_epoch_funcs(
        algo_class_in=SeqwiseAlgoSplitHorizonExplCollection,
        replay_buffer=replay_buffer,
        expl_step_collector=expl_step_collector,
        eval_path_collector=eval_path_collector,
        seq_eval_collector=seq_eval_collector,
        diagno_writer=diagno_writer,
        eval_policy=eval_policy,
        df=df,
        config=config,
        expl_env=expl_env,
        eval_env=eval_env,
        trainer=trainer,
    )
    algorithm.to(ptu.device)

    return algorithm


if __name__ == "__main__":
    config, config_path_name = parse_args(
        default="config/all_in_one_config/hopper/"
                "rnn_v2.yaml",
        return_config_path_name=True,
    )
    config = prepare_hparams(config)

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
    ptu.set_gpu_mode(config.gpu)  # optionally set the GPU (default=False)

    algorithm = create_experiment(config, config_path_name)
    algorithm.train()
