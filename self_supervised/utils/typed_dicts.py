import numpy as np
from prodict import Prodict
import torch


class AlgoKwargsMapping(Prodict):
    num_epochs: int
    def __init__(self,
                 num_epochs: int):
        super().__init__(
            num_epochs=num_epochs
        )


class TrainerKwargsMapping(Prodict):
    discount: float
    def __init__(self,
                 discount: float):
        super().__init__(
            discount=discount
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
    info_loss_kwargs: InfoLossParamsMapping

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
                 info_loss_kwargs: InfoLossParamsMapping
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
            info_loss_kwargs=info_loss_kwargs,
            leaky_slope=leaky_slope,
        )


class VariantMapping(Prodict):
    algorithm: str
    version: str
    hidden_layer: list
    replay_buffer_size: int
    skill_dim: int
    seq_len: int
    batch_size: int
    layer_norm: bool
    mode_latent_kwargs: ModeLatentKwargsMapping
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
                 batch_size: int,
                 num_epochs: int,
                 layer_norm: bool,
                 mode_latent_kwargs: ModeLatentKwargsMapping,
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
            batch_size=batch_size,
            num_epochs=num_epochs,
            layer_norm=layer_norm,
            mode_latent_kwargs=mode_latent_kwargs,
            env_kwargs=env_kwargs,
            algo_kwargs=algo_kwargs,
            trainer_kwargs=trainer_kwargs,
        )


class TransitionMapping(Prodict):
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
                 **kwargs):

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
                 **kwargs):

        Prodict.__init__(
            self,
            obs_seqs=obs,
            action_seqs=action,
            reward_seqs=reward,
            terminal_seqs=terminal,
            next_obs_seqs=next_obs,
            mode=mode,
            **kwargs
           )
