from prodict import Prodict


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
    rnn_dim: int
    num_rnn_layers: int
    rnn_dropout: float
    hidden_units_mode_encoder: list
    hidden_units_action_decoder: list
    num_mode_repeat: int
    std_decoder: float
    device: torch.device
    leaky_slope: float

    def __init__(self,
                 rnn_dim: int,
                 num_rnn_layers: int,
                 rnn_dropout: float,
                 hidden_units_mode_encoder: list,
                 hidden_units_action_decoder: list,
                 num_mode_repeat: int,
                 std_decoder: float,
                 device: torch.device,
                 leaky_slope: float,
                 ):
        super().__init__(
            rnn_dim=rnn_dim,
            num_rnn_layers=num_rnn_layers,
            rnn_dropout=rnn_dropout,
            hidden_units_mode_encoder=hidden_units_mode_encoder,
            hidden_units_action_decoder=hidden_units_action_decoder,
            num_mode_repeat=num_mode_repeat,
            std_decoder=std_decoder,
            device=device,
            leaky_slope=leaky_slope,
        )
