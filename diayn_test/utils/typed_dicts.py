from prodict import Prodict


import self_supervised.utils.typed_dicts as tdss


class AlgoKwargsDiaynMapping(Prodict):
    batch_size: int
    num_epochs: int
    seq_len: int
    num_eval_steps_per_epoch: int
    num_expl_steps_per_train_loop: int
    num_trains_per_train_loop: int
    num_train_loops_per_epoch: int
    min_num_steps_before_training: int

    def __init__(self,
                 batch_size: int,
                 num_epochs: int,
                 seq_len: int,
                 num_eval_steps_per_epoch: int,
                 num_expl_steps_per_train_loop: int,
                 num_trains_per_train_loop: int,
                 num_train_loops_per_epoch: int,
                 min_num_steps_before_training: int,
                 ):
        super().__init__(
            batch_size=batch_size,
            num_epochs=num_epochs,
            seq_len=seq_len,
            num_eval_steps_per_epoch=num_eval_steps_per_epoch,
            num_expl_steps_per_train_loop=num_expl_steps_per_train_loop,
            num_trains_per_train_loop=num_trains_per_train_loop,
            num_train_loops_per_epoch=num_train_loops_per_epoch,
            min_num_steps_before_training=min_num_steps_before_training
        )


class VariantMapping(Prodict):
    algorithm: str
    version: str
    hidden_layer: list
    replay_buffer_size: int
    skill_dim: int
    layer_norm: bool
    env_kwargs: tdss.EnvKwargsMapping
    algo_kwargs: AlgoKwargsDiaynMapping
    trainer_kwargs: tdss.TrainerKwargsMapping

    def __init__(self,
                 algorithm: str,
                 version: str,
                 hidden_layer: list,
                 replay_buffer_size: int,
                 skill_dim: int,
                 layer_norm: bool,
                 env_kwargs: tdss.EnvKwargsMapping,
                 algo_kwargs: AlgoKwargsDiaynMapping,
                 trainer_kwargs: tdss.TrainerKwargsMapping):
        super().__init__(
            algorithm=algorithm,
            version=version,
            hidden_layer=hidden_layer,
            replay_buffer_size=replay_buffer_size,
            skill_dim=skill_dim,
            layer_norm=layer_norm,
            env_kwargs=env_kwargs,
            algo_kwargs=algo_kwargs,
            trainer_kwargs=trainer_kwargs,
        )