import numpy as np
import torch
import copy

from rlkit.launchers.launcher_util import setup_logger
from rlkit.torch.networks import FlattenMlp
import rlkit.torch.pytorch_util as ptu
from rlkit.torch.sac.diayn.diayn_env_replay_buffer import DIAYNEnvReplayBuffer
from rlkit.samplers.data_collector.path_collector import MdpPathCollector
from rlkit.samplers.data_collector.step_collector import MdpStepCollector
from rlkit.torch.sac.diayn.diayn_path_collector import DIAYNMdpPathCollector

from mode_disent_no_ssm.utils.parse_args import parse_args

from latent_with_splitseqs.config.fun.get_env import get_env
from latent_with_splitseqs.config.fun.get_obs_dims_used_policy \
    import get_obs_dims_used_policy
from latent_with_splitseqs.config.fun.get_obs_dims_used_df \
    import get_obs_dims_used_df
from latent_with_splitseqs.config.fun.get_diagnostics_writer import get_diagnostics_writer

from my_utils.dicts.get_config_item import get_config_item

from code_slac.network.latent import Gaussian

from diayn_cont.trainer.diayn_cont_trainer import DIAYNContTrainer
from diayn_cont.policy.skill_policy_skill_selector_obs_dim_select \
    import SkillTanhGaussianPolicyWithSkillSelectorObsSelect
from diayn_cont.policy.skill_policy_with_skill_selector import MakeDeterministic
from diayn_cont.algo.cont_algo import DIAYNContAlgo
from diayn_cont.data_collector.seq_eval_collector import MdpPathCollectorWithReset
from diayn_cont.post_epoch_funcs.get_algo import get_algo


def create_experiment(config, config_path_name):
    expl_env = get_env(**config.env_kwargs)
    eval_env = copy.deepcopy(expl_env)

    obs_dim = expl_env.observation_space.low.size
    action_dim = expl_env.action_space.low.size
    used_obs_dims_policy = get_obs_dims_used_policy(
        obs_dim=obs_dim,
        config=config,
    )

    scripts_to_copy = get_config_item(
        config,
        key="scripts_to_copy",
        default=None,
    )

    seed = get_config_item(
        config,
        key="seed",
        default=0,
    )
    torch.manual_seed = seed
    expl_env.seed(seed)
    eval_env.seed(seed)
    np.random.seed(seed)

    M = config["layer_size"]
    skill_dim = config["skill_dim"]
    qf1 = FlattenMlp(
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

    policy = SkillTanhGaussianPolicyWithSkillSelectorObsSelect(
        hidden_sizes=[M, M],
        obs_dim=len(used_obs_dims_policy),
        action_dim=action_dim,
        obs_dims_selected=used_obs_dims_policy,
        obs_dim_real=obs_dim
    )
    eval_policy = MakeDeterministic(policy)
    expl_step_collector = MdpStepCollector(
        env=expl_env,
        policy=policy,
    )
    eval_path_collector = MdpPathCollector(
        env=eval_env,
        policy=eval_policy,
        max_num_epoch_paths_saved=100,
    )
    post_epoch_eval_path_collector = MdpPathCollectorWithReset(
        env=eval_env,
        policy=eval_policy,
        max_num_epoch_paths_saved=100,
    )
    obs_dims_used_df = get_obs_dims_used_df(
        obs_dim=obs_dim,
        obs_dims_used=config["df_kwargs"]["obs_dims_used"],
        obs_dims_used_except=config["df_kwargs"]["obs_dims_used_except"]
    )
    df = Gaussian(
        input_dim=obs_dims_used_df,
        output_dim=skill_dim,
        hidden_units=[M, M],
        leaky_slope=0.1,
        std=config["df_kwargs"]["std"],
    )
    trainer = DIAYNContTrainer(
        env=expl_env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        df=df,
        **config["trainer_kwargs"]
    )
    replay_buffer = DIAYNEnvReplayBuffer(
        max_replay_buffer_size=config["max_replay_buffer_size"],
        env=eval_env,
        skill_dim=skill_dim,
    )
    diagno_writer = get_diagnostics_writer(
        run_comment=config['run_comment'],
        config=config,
        scripts_to_copy=scripts_to_copy,
        seed=seed,
        config_path_name=config_path_name,
    )
    algo_kwargs = dict(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=expl_env,
        eval=eval_env,
        exploration_data_collector=expl_step_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        **config['algo_kwargs'],
    )
    algorithm = get_algo(
        algo_class=DIAYNContAlgo,
        algo_kwargs=algo_kwargs,
        diagnostic_writer=diagno_writer,
        eval_policy=eval_policy,
        post_epoch_eval_path_collector=post_epoch_eval_path_collector,
        config=config,
    )

    return algorithm


if __name__ == "__main__":
    config, config_path_name = parse_args(
        default="config/config_files/twoDnav.yaml",
        return_config_path_name=True
    )

    setup_logger(
        config.algorithm + config.version,
    )
    ptu.set_gpu_mode(config.gpu)

    algorithm = create_experiment(config, config_path_name)
    algorithm.train()