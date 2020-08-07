import numpy as np
from prodict import Prodict
import torch
import sys


class AlgoKwargsMapping(Prodict):
    batch_size: int
    num_epochs: int
    num_eval_steps_per_epoch: int
    num_trains_per_expl_step: int
    num_mode_trains_per_train_step: int
    num_sac_trains_per_train_step: int
    num_train_loops_per_epoch: int
    min_num_steps_before_training: int
    def __init__(self,
                 batch_size: int,
                 num_epochs: int,
                 num_eval_steps_per_epoch: int,
                 num_trains_per_expl_step: int,
                 num_train_loops_per_epoch: int,
                 num_sac_trains_per_train_step: int,
                 num_mode_trains_per_train_step: int,
                 min_num_steps_before_training: int,
                 ):
        super().__init__(
            batch_size=batch_size,
            num_epochs=num_epochs,
            num_eval_steps_per_epoch=num_eval_steps_per_epoch,
            num_trains_per_expl_step=num_trains_per_expl_step,
            num_train_loops_per_epoch=num_train_loops_per_epoch,
            num_sac_trains_per_train_step=num_sac_trains_per_train_step,
            num_mode_trains_per_train_step=num_mode_trains_per_train_step,
            min_num_steps_before_training=min_num_steps_before_training,
        )


class TrainerKwargsMapping(Prodict):
    discount: float
    soft_target_tau: float
    target_update_period: int
    policy_lr: float
    qf_lr: float
    reward_scale: int
    use_automatic_entropy_tuning: float
    def __init__(self,
                 discount: float,
                 soft_target_tau: float,
                 target_update_period: int,
                 policy_lr: float,
                 qf_lr: float,
                 reward_scale: int,
                 use_automatic_entropy_tuning: float,):
        super().__init__(
            discount=discount,
            soft_target_tau=soft_target_tau,
            target_update_period=target_update_period,
            policy_lr=policy_lr,
            qf_lr=qf_lr,
            reward_scale=reward_scale,
            use_automatic_entropy_tuning=use_automatic_entropy_tuning
        )


class EnvKwargsMapping(Prodict):
    gym_id: str
    action_repeat: int
    normalize_states: bool

    def __init__(self,
                 gym_id: str,
                 action_repeat: int,
                 normalize_states: bool):
        super().__init__(
            gym_id=gym_id,
            action_repeat=action_repeat,
            normalize_states=normalize_states
        )

    def init(self):
        self.normalize_states = True


class InfoLossParamsMapping(Prodict):
    alpha: float
    lamda: float
    def __init__(self,
                 alpha: float,
                 lamda: float
                 ):
        super().__init__(
            alpha=alpha,
            lamda=lamda
        )


class ModeLatentKwargsMapping(Prodict):
    feature_dim: int
    rnn_dim: int
    num_rnn_layers: int
    rnn_dropout: float
    hidden_units_mode_encoder: list
    hidden_units_action_decoder: list
    num_mode_repeat: int
    std_decoder: float
    leaky_slope: float

    def __init__(self,
                 feature_dim: int,
                 rnn_dim: int,
                 num_rnn_layers: int,
                 rnn_dropout: float,
                 hidden_units_mode_encoder: list,
                 hidden_units_action_decoder: list,
                 num_mode_repeat: int,
                 std_decoder: float,
                 leaky_slope: float,
                 ):
        super().__init__(
            feature_dim=feature_dim,
            rnn_dim=rnn_dim,
            num_rnn_layers=num_rnn_layers,
            rnn_dropout=rnn_dropout,
            hidden_units_mode_encoder=hidden_units_mode_encoder,
            hidden_units_action_decoder=hidden_units_action_decoder,
            num_mode_repeat=num_mode_repeat,
            std_decoder=std_decoder,
            leaky_slope=leaky_slope,
        )


class VariantMapping(Prodict):
    algorithm: str
    version: str
    hidden_layer: list
    replay_buffer_size: int
    skill_dim: int
    seq_len: int
    layer_norm: bool
    mode_latent_kwargs: ModeLatentKwargsMapping
    info_loss_kwargs: InfoLossParamsMapping
    env_kwargs: EnvKwargsMapping
    algo_kwargs: AlgoKwargsMapping
    trainer_kwargs: TrainerKwargsMapping

    def __init__(self,
                 algorithm: str,
                 version: str,
                 hidden_layer: list,
                 replay_buffer_size: int,
                 skill_dim: int,
                 seq_len: int,
                 layer_norm: bool,
                 mode_latent_kwargs: ModeLatentKwargsMapping,
                 info_loss_kwargs: InfoLossParamsMapping,
                 env_kwargs: EnvKwargsMapping,
                 algo_kwargs: AlgoKwargsMapping,
                 trainer_kwargs: TrainerKwargsMapping):
        super().__init__(
            algorithm=algorithm,
            version=version,
            hidden_layer=hidden_layer,
            replay_buffer_size=replay_buffer_size,
            skill_dim=skill_dim,
            seq_len=seq_len,
            layer_norm=layer_norm,
            mode_latent_kwargs=mode_latent_kwargs,
            info_loss_kwargs=info_loss_kwargs,
            env_kwargs=env_kwargs,
            algo_kwargs=algo_kwargs,
            trainer_kwargs=trainer_kwargs,
        )


class SlicableProdict(Prodict):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._iter_counter = 0

    def __getitem__(self, item):
        if type(item) is str:
            return super().__getitem__(item)

        else:
            return self._get_constr()(
                **{
                    # TODO: use hasattr in stead type checking for torch.Tensor or np.ndarray
                    k: v[item]
                    if self._is_array_type(v) else v
                    for (k, v) in self.items()
                }
            )

    def transpose(self, *args, **kwargs):
        return self._get_constr()(
            **{
                #TODO: use hasattr in stead type checking for torch.Tensor or np.ndarray
                k: v.transpose(*args, **kwargs)
                if self._is_array_type(v) else v
                for k, v in self.items()
            }
        )

    def permute(self, *args, **kwargs):
        return self._get_constr()(
            **{
                #TODO: use hasattr in stead type checking for torch.Tensor or np.ndarray
                k: v.permute(*args, **kwargs)
                if self._is_array_type(v) else v
                for k, v in self.items()
            }
        )

    def reshape(self, *args, **kwargs):
        return self._get_constr()(
            **{
                k: v.reshape(*args, **kwargs)
                if self._is_array_type(v) else v
                for k, v in self.items()
            }
        )

    def view(self, *args, **kwargs):
        return self._get_constr()(
            **{
                k: v.view(*args, **kwargs)
                if self._is_array_type(v) else v
                for k, v in self.items()
            }
        )

    def _get_constr(self):
        constr = getattr(sys.modules[__name__], self.__class__.__name__)
        return constr

    @property
    def size_first_dim(self):
        sizes_first_dim = [v.shape[0] for v in self.values() if self._is_array_type(v)]

        if all(sizes_first_dim[0]== size for size in sizes_first_dim):
            return sizes_first_dim[0]
        else:
            raise ValueError("Arrays in the Mapping don't match on the first dimension")

    def __iter__(self):
        return self

    def __next__(self):
        #for idx in range(self.size_first_dim):
        #    return self[idx]
        if self._iter_counter < self.size_first_dim:
            self._iter_counter += 1
            return self[self._iter_counter - 1]
        else:
            raise StopIteration

    def _is_array_type(self, to_check):
        return type(to_check) in [torch.Tensor, np.ndarray]


class TransitionMapping(SlicableProdict):
    obs: np.ndarray
    action: np.ndarray
    reward: np.ndarray
    terminal: np.ndarray
    next_obs: np.ndarray

    def __init__(self,
                 obs: np.ndarray,
                 action: np.ndarray,
                 reward: np.ndarray,
                 terminal: np.ndarray,
                 next_obs: np.ndarray,
                 **kwargs
                 ):

        super(TransitionMapping, self).__init__(
            obs=obs,
            action=action,
            reward=reward,
            terminal=terminal,
            next_obs=next_obs,
            **kwargs
        )


class TransitionModeMapping(TransitionMapping):
    mode: np.ndarray

    def __init__(self,
                 obs: np.ndarray,
                 action: np.ndarray,
                 reward: np.ndarray,
                 terminal: np.ndarray,
                 next_obs: np.ndarray,
                 mode: np.ndarray,
                 **kwargs
                 ):

        super(TransitionModeMapping, self).__init__(
            obs=obs,
            action=action,
            reward=reward,
            terminal=terminal,
            next_obs=next_obs,
            mode=mode,
            **kwargs
           )


class TransitionMappingTorch(SlicableProdict):
    obs: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor
    terminal: torch.Tensor
    next_obs: torch.Tensor

    def __init__(self,
                 obs: torch.Tensor,
                 action: torch.Tensor,
                 reward: torch.Tensor,
                 terminal: torch.Tensor,
                 next_obs: torch.Tensor,
                 ):
        super(TransitionMappingTorch, self).__init__(
            obs=obs,
            action=action,
            reward=reward,
            terminal=terminal,
            next_obs=next_obs,
        )


class TransitionModeMappingTorch(SlicableProdict):
    obs: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor
    terminal: torch.Tensor
    next_obs: torch.Tensor
    mode: torch.Tensor

    def __init__(self,
                 obs: torch.Tensor,
                 action: torch.Tensor,
                 reward: torch.Tensor,
                 terminal: torch.Tensor,
                 next_obs: torch.Tensor,
                 mode: torch.Tensor,
                 **kwargs
                 ):
        super(TransitionModeMappingTorch, self).__init__(
            obs=obs,
            action=action,
            reward=reward,
            terminal=terminal,
            next_obs=next_obs,
            mode=mode,
            **kwargs
        )

    def get_transition_mapping(self) -> TransitionMappingTorch:
        return TransitionMappingTorch(
            obs=self.obs,
            action=self.action,
            reward=self.reward,
            terminal=self.terminal,
            next_obs=self.next_obs
        )


# TODO: Nest Mapping into TanhGaussianPolicy
class ActionMapping(Prodict):
    action: np.ndarray
    agent_info: dict

    def __init__(self,
                 action: np.ndarray,
                 agent_info: dict):
        super().__init__(
            action=action,
            agent_info=agent_info
        )


class ForwardReturnMapping(Prodict):
    action: torch.Tensor
    mean: torch.Tensor
    log_std: torch.Tensor
    log_prob: torch.Tensor
    entropy: torch.Tensor
    std: torch.Tensor
    mean_action_log_prob: torch.Tensor
    pre_tanh_value: torch.Tensor

    def __init__(self,
                 action: torch.Tensor,
                 mean: torch.Tensor,
                 log_std: torch.Tensor,
                 log_prob: torch.Tensor,
                 entropy: torch.Tensor,
                 std: torch.Tensor,
                 mean_action_log_prob: torch.Tensor,
                 pre_tanh_value: torch.Tensor):
        super().__init__()
        self.action = action
        self.mean = mean
        self.log_std = log_std
        self.log_prob = log_prob
        self.entropy = entropy
        self.std = std
        self.mean_action_log_prob = mean_action_log_prob
        self.pre_tanh_value = pre_tanh_value


class TransitonModeMappingDiscreteSkills(TransitionModeMapping):
    skill_id: np.ndarray

    def __init__(self,
                 obs: np.ndarray,
                 action: np.ndarray,
                 reward: np.ndarray,
                 terminal: np.ndarray,
                 next_obs: np.ndarray,
                 mode: np.ndarray,
                 skill_id: np.ndarray,
                 **kwargs
                 ):

        Prodict.__init__(
            self,
            obs=obs,
            action=action,
            reward=reward,
            terminal=terminal,
            next_obs=next_obs,
            mode=mode,
            skill_id=skill_id,
            **kwargs
        )