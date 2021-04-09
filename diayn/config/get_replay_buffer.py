from my_utils.dicts.get_config_item import get_config_item

from rlkit.torch.sac.diayn.diayn_env_replay_buffer import DIAYNEnvReplayBuffer

from diayn_cont.memory.replay_buffer import DIAYNContEnvReplayBuffer

from diayn.memory.replay_buffer_prioritized import DIAYNEnvReplayBufferEBP
from diayn.energy.calc_energy_1D_pos_dim import calc_energy_1d_pos_dim
from diayn.energy.calc_energy_mcar import calc_energy_mcar


def get_replay_buffer(
        config: dict,
        replay_buffer_kwargs
) -> DIAYNEnvReplayBuffer:
    ebp_sampling = get_config_item(
        config=config,
        key='ebp_sampling',
        default=False
    )

    if config['env_kwargs']['env_id'] == "MountainCarContinuous-v0":
        energy_fun = calc_energy_mcar

    else:
        energy_fun = calc_energy_1d_pos_dim

    if ebp_sampling:
        replay_buffer = DIAYNEnvReplayBufferEBP(
            calc_path_energy_fun=energy_fun,
            **replay_buffer_kwargs
        )

    else:
        replay_buffer = DIAYNContEnvReplayBuffer(
            **replay_buffer_kwargs
        )

    return replay_buffer
