import numpy as np


def calc_energy_1d_pos_dim(
        path: dict,
        g=9.81,
        m=1,
        inertia=1,
        max_transition_energy=1,):
    obs_dim = path['observations'].shape[-1]
    assert obs_dim % 2 == 0
    height_dim = 1
    velocity_dim = obs_dim // 2
    angular_velocity_dim = velocity_dim + 1

    transition_energy = {}
    transition_energy['pot'] = np.zeros((path['observations'].shape[0],))
    transition_energy['kin'] = np.zeros((path['observations'].shape[0],))
    transition_energy['rot'] = np.zeros((path['observations'].shape[0],))

    E_pot_before = m * g * path['observations'][0, height_dim]
    E_kin_before = 0.5 * m * np.power(path['observations'][0, velocity_dim], 2)
    E_rot_before = 0.5 * inertia * np.power(path['observations'][0, angular_velocity_dim], 2)

    for idx, obs in enumerate(path['next_observations']):
        E_pot = m * g * obs[height_dim]
        E_kin = 0.5 * m * np.power(obs[velocity_dim], 2)
        E_rot = 0.5 * inertia * np.power(obs[angular_velocity_dim], 2)

        transition_energy['pot'][idx] = np.clip(E_pot - E_pot_before, 0, max_transition_energy)
        transition_energy['kin'][idx] = np.clip(E_kin - E_kin_before, 0, max_transition_energy)
        transition_energy['rot'][idx] = np.clip(E_rot - E_rot_before, 0, max_transition_energy)

        E_pot_before = E_pot
        E_kin_before = E_kin
        E_rot_before = E_rot

    path_energy = dict(
        pot=np.sum(transition_energy['pot']),
        kin=np.sum(transition_energy['kin']),
        rot=np.sum(transition_energy['rot']),
    )

    return path_energy
