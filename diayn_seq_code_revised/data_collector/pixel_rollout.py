import numpy as np

from self_supervised.env_wrapper.pixel_wrapper import PixelNormalizedBoxEnvWrapper

def pixel_rollout(
        env: PixelNormalizedBoxEnvWrapper,
        agent,
        max_path_length=np.inf,
        render=False,
        render_kwargs=None,
):
    """
    The following value for the following keys will be a 2D array, with the
    first dimension corresponding to the time dimension.
     - observations
     - actions
     - rewards
     - next_observations
     - terminals

    The next two elements will be lists of dictionaries, with the index into
    the list being the index into the time
     - agent_infos
     - env_infos
    """
    observations = []
    pixel_obs = []
    actions = []
    rewards = []
    terminals = []
    agent_infos = []
    env_infos = []
    o = env.reset()
    o = o['state_obs']
    agent.reset()
    next_o = None
    path_length = 0
    if render:
        env.render(**render_kwargs)
    while path_length < max_path_length:

        a, agent_info = agent.get_action(o)
        next_o, r, d, env_info = env.step(a)

        observations.append(next_o['state_obs'])
        pixel_obs.append(next_o['pixel_obs'])

        rewards.append(r)
        terminals.append(d)
        actions.append(a)
        agent_infos.append(agent_info)
        env_infos.append(env_info)

        path_length += 1

        if max_path_length == np.inf and d:
            break
        o = next_o['state_obs']
        if render:
            env.render(**render_kwargs)

    actions = np.array(actions)
    if len(actions.shape) == 1:
        actions = np.expand_dims(actions, 1)
    observations = np.array(observations)
    next_o = next_o['state_obs']
    if len(observations.shape) == 1:
        observations = np.expand_dims(observations, 1)
        next_o = np.array([next_o['state_obs']])
    next_observations = np.vstack(
        (
            observations[1:, :],
            np.expand_dims(next_o, 0)
        )
    )
    pixel_obs = np.stack(pixel_obs, axis=0)
    return dict(
        observations=observations,
        actions=actions,
        rewards=np.array(rewards).reshape(-1, 1),
        next_observations=(next_observations, pixel_obs),
        terminals=np.array(terminals).reshape(-1, 1),
        agent_infos=agent_infos,
        env_infos=env_infos,
    )
