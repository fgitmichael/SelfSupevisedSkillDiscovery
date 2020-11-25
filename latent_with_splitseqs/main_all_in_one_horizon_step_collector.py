import torch
import math
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

from seqwise_cont_skillspace.networks.contant_uniform import ConstantUniformMultiDim
from seqwise_cont_skillspace.data_collector.skill_selector_cont_skills import \
    SkillSelectorContinous
from seqwise_cont_skillspace.utils.info_loss import GuidedInfoLoss

from mode_disent_no_ssm.utils.parse_args import parse_args, parse_args_hptuning

from latent_with_splitseqs.data_collector.seq_collector_split import SeqCollectorSplitSeq
from latent_with_splitseqs.config.fun.get_env import get_env
from latent_with_splitseqs.config.fun.get_obs_dims_used_policy \
    import get_obs_dims_used_policy
from latent_with_splitseqs.config.fun.get_df_and_trainer import get_df_and_trainer
from latent_with_splitseqs.config.fun.get_feature_dim_obs_dim \
    import get_feature_dim_obs_dim
from latent_with_splitseqs.utils.loglikelihoodloss import GuidedKldLogOnlyLoss
from latent_with_splitseqs.algo.algo_latent_splitseqs import \
    SeqwiseAlgoRevisedSplitSeqs
from latent_with_splitseqs.config.fun.get_skill_prior import get_skill_prior
from latent_with_splitseqs.config.fun.get_loss_fun import get_loss_fun
from latent_with_splitseqs.data_collector.seq_collector_over_horizon \
    import SeqCollectorHorizon
from latent_with_splitseqs.algo.algo_latent_split_horizon_expl_collection \
    import SeqwiseAlgoSplitHorizonExplCollection
from latent_with_splitseqs.config.fun.get_algo import get_algo_with_post_epoch_funcs
from latent_with_splitseqs.config.fun.get_replay_buffer import get_replay_buffer
from latent_with_splitseqs.config.fun.get_diagnostics_writer import get_diagnostics_writer
from latent_with_splitseqs.config.fun.get_random_hp_params import get_random_hp_params
from latent_with_splitseqs.config.fun.prepare_hparams import prepare_hparams

from diayn_original_tb.algo.algo_diayn_tb import DIAYNTorchOnlineRLAlgorithmTb


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

    sep_str = " | "
    run_comment = sep_str
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
    expl_step_collector = SeqCollectorHorizon(
        expl_env,
        policy,
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
    replay_buffer = get_replay_buffer(
        config=config,
        env=expl_env,
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
    config, config_path_name = parse_args_hptuning(
        default="config/all_in_one_config/halfcheetah/"
                "rnn_v2.yaml",
        default_min="./config/all_in_one_config/mountaincar/"
                    "random_hp_search/"
                    "srnn_v0_min.yaml",
        default_max="./config/all_in_one_config/mountaincar/"
                    "random_hp_search/"
                    "srnn_v0_max.yaml",
        default_hp_tuning=False,
        return_config_path_name=True,
    )

    if config.random_hp_tuning:
        config = get_random_hp_params(config)
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
    ptu.set_gpu_mode(False)  # optionally set the GPU (default=False)

    algorithm = create_experiment(config, config_path_name)
    algorithm.train()
