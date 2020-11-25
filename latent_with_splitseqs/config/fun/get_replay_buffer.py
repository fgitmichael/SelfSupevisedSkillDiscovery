from latent_with_splitseqs.memory.replay_buffer_for_latent import LatentReplayBuffer

from self_supervised.base.replay_buffer.replay_buffer_base import SequenceReplayBuffer


def get_replay_buffer(config, env) -> SequenceReplayBuffer:
    replay_buffer = LatentReplayBuffer(
        max_replay_buffer_size=config.replay_buffer_size,
        seq_len=config.seq_len,
        mode_dim=config.skill_dim,
        env=env,
    )
    return replay_buffer

