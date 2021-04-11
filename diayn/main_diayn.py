import copy
import warnings
import torch
import numpy as np

from mode_disent_no_ssm.utils.parse_args import parse_args

from rlkit.launchers.launcher_util import setup_logger
from rlkit.torch.networks import FlattenMlp
import rlkit.torch.pytorch_util as ptu
from rlkit.samplers.data_collector.path_collector import MdpPathCollector
from rlkit.samplers.data_collector.step_collector import MdpStepCollector
from rlkit.torch.sac.diayn.diayn import DIAYNTrainer

from latent_with_splitseqs.config.fun.get_env import get_env
from latent_with_splitseqs.config.fun.get_obs_dims_used_policy \
    import get_obs_dims_used_policy
from latent_with_splitseqs.config.fun.get_diagnostics_writer import get_diagnostics_writer

from my_utils.dicts.get_config_item import get_config_item

from diayn.policies.diayn_policy import \
    SkillTanhGaussianPolicyObsSelectDIAYN, MakeDeterministic
from diayn.config.get_algo import get_algo
from diayn.algo.diayn_algo import DIAYNAlgo
from diayn.classifier.diayn_classifier_obs_dim_select \
    import FlattenMlpInputDimSelect
from diayn.config.get_replay_buffer import get_replay_buffer

from diayn_cont.data_collector.seq_eval_collector import MdpPathCollectorWithReset

from latent_with_splitseqs.config.fun.get_obs_dims_used_df \
    import get_obs_dims_used_df


def create_experiment(config, config_path_name):
    expl_env = get_env(**config.env_kwargs)
    eval_env = copy.deepcopy(expl_env)

    obs_dim = expl_env.observation_space.low.size
    action_dim = expl_env.action_space.low.size
    used_obs_dims_policy = get_obs_dims_used_policy(
        obs_dim=obs_dim,
        config=config,
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

    policy = SkillTanhGaussianPolicyObsSelectDIAYN(
        hidden_sizes=[M, M],
        skill_dim=skill_dim,
        obs_dim=len(used_obs_dims_policy) + skill_dim,
        action_dim=action_dim,
        obs_dims_selected=used_obs_dims_policy,
        obs_dim_real=obs_dim,
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
    df = FlattenMlpInputDimSelect(
        input_size=len(obs_dims_used_df),
        used_dims=obs_dims_used_df,
        output_size=skill_dim,
        hidden_sizes=[M, M],
    )
    trainer = DIAYNTrainer(
        env=eval_env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        df=df,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        **config['trainer_kwargs']
    )
    replay_buffer_kwargs = dict(
        max_replay_buffer_size=config["replay_buffer_size"],
        env=eval_env,
        skill_dim=skill_dim,
    )
    replay_buffer = get_replay_buffer(
        replay_buffer_kwargs=replay_buffer_kwargs,
        config=config,
    )
    diagno_writer = get_diagnostics_writer(
        run_comment=" ".join((config['algorithm'], config['version'])),
        config=config,
        scripts_to_copy=config['test_script_path'],
        seed=seed,
        config_path_name=config_path_name,
    )
    algo_kwargs = dict(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_step_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        diagnostic_writer=diagno_writer,
        **config['algorithm_kwargs'],
    )
    algorithm = get_algo(
        algo_class=DIAYNAlgo,
        algo_kwargs=algo_kwargs,
        df=df,
        diagnostic_writer=diagno_writer,
        eval_policy=eval_policy,
        post_epoch_eval_path_collector=post_epoch_eval_path_collector,
        config=config,
    )
    algorithm.to(ptu.device)

    return algorithm


if __name__ == "__main__":
    config, config_path_name = parse_args(
        default="config/config_files/hopper_ebp.yaml",
        return_config_path_name=True
    )

    setup_logger(
        config.algorithm + config.version,
        )
    ptu.set_gpu_mode(config.gpu)
    if not config.gpu:
        warnings.warn("GPU is deactivated")

    algorithm = create_experiment(config, config_path_name)
    algorithm.train()
