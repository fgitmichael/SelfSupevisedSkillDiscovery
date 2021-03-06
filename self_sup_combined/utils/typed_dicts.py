from prodict import Prodict
import torch

import self_supervised.utils.typed_dicts as tdss


class ModeTrainerDataMapping(tdss.SlicableProdict):
    skills_gt: torch.Tensor
    obs_seq: torch.Tensor

    def __init__(self,
                 skills_gt: torch.Tensor,
                 obs_seq: torch.Tensor):
        super().__init__(
            skills_gt=skills_gt,
            obs_seq=obs_seq
        )


class ModeEncoderKwargsMapping(Prodict):
    feature_dim: int
    rnn_dim: int
    num_rnn_layers: int
    rnn_dropout: float
    hidden_units: list

    def __init__(self,
                 feature_dim: int,
                 rnn_dim: int,
                 num_rnn_layers: int,
                 rnn_dropout: float,
                 hidden_units: list,
                 ):
        super().__init__(
            feature_dim=feature_dim,
            rnn_dim=rnn_dim,
            num_rnn_layers=num_rnn_layers,
            rnn_dropout=rnn_dropout,
            hidden_units=hidden_units,
        )


class VariantMapping(Prodict):
    algorithm: str
    version: str
    hidden_layer: list
    replay_buffer_size: int
    skill_dim: int
    seq_len: int
    layer_norm: bool
    mode_encoder_kwargs: ModeEncoderKwargsMapping
    info_loss_kwargs: tdss.InfoLossParamsMapping
    env_kwargs: tdss.EnvKwargsMapping
    algo_kwargs: tdss.AlgoKwargsMapping
    trainer_kwargs: tdss.TrainerKwargsMapping

    def __init__(self,
                 algorithm: str,
                 version: str,
                 hidden_layer: list,
                 replay_buffer_size: int,
                 skill_dim: int,
                 seq_len: int,
                 layer_norm: bool,
                 mode_encoder_kwargs: ModeEncoderKwargsMapping,
                 info_loss_kwargs: tdss.InfoLossParamsMapping,
                 env_kwargs: tdss.EnvKwargsMapping,
                 algo_kwargs: tdss.AlgoKwargsMapping,
                 trainer_kwargs: tdss.TrainerKwargsMapping):
        super().__init__(
            algorithm=algorithm,
            version=version,
            hidden_layer=hidden_layer,
            replay_buffer_size=replay_buffer_size,
            skill_dim=skill_dim,
            seq_len=seq_len,
            layer_norm=layer_norm,
            mode_encoder_kwargs=mode_encoder_kwargs,
            info_loss_kwargs=info_loss_kwargs,
            env_kwargs=env_kwargs,
            algo_kwargs=algo_kwargs,
            trainer_kwargs=trainer_kwargs,
        )


