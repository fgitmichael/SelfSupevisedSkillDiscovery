from typing import Tuple

from self_supervised.base.replay_buffer.replay_buffer_base import SequenceReplayBuffer

from latent_with_splitseqs.memory.replay_buffer_for_latent import LatentReplayBuffer
from latent_with_splitseqs.memory.replay_buffer_latent_splitseq_sampling_fixed_seqlen \
    import LatentReplayBufferSplitSeqSamplingFixedSeqLen
from latent_with_splitseqs.memory.replay_buffer_latent_splitseq_sampling_random_seqlen \
    import LatentReplayBufferSplitSeqSamplingRandomSeqLen
from latent_with_splitseqs.data_collector.seq_collector_over_horizon_splitseq_save \
    import SeqCollectorHorizonSplitSeqSaving
from latent_with_splitseqs.data_collector.seq_collector_over_horizon_wholeseq_save \
    import SeqCollectorHorizonWholeSeqSaving

from diayn_seq_code_revised.base.data_collector_base import DataCollectorRevisedBase

from my_utils.dicts.get_config_item import get_config_item


def get_replay_buffer_and_expl_collector(
        config,
        expl_env,
        policy,
        skill_selector
) -> Tuple[SequenceReplayBuffer, DataCollectorRevisedBase]:
    variant = get_config_item(
        config=config,
        key='replay_seq_sampling',
        default='fixed',
    )

    if variant == 'fixed':
        expl_step_collector = SeqCollectorHorizonSplitSeqSaving(
            expl_env,
            policy,
            max_seqs=get_config_item(config, 'max_seqs', 5000),
            skill_selector=skill_selector,
        )
        replay_buffer = LatentReplayBuffer(
            max_replay_buffer_size=config.replay_buffer_size,
            seq_len=config.seq_len,
            mode_dim=config.skill_dim,
            env=expl_env,
        )

    elif variant == 'sampling':
        expl_step_collector = SeqCollectorHorizonWholeSeqSaving(
            expl_env,
            policy,
            max_seqs=get_config_item(config, 'max_seqs', 5000),
            skill_selector=skill_selector,
        )
        replay_buffer = LatentReplayBufferSplitSeqSamplingFixedSeqLen(
            max_replay_buffer_size=config.replay_buffer_size,
            seq_len=config.horizon_len, # Now whole horizon is saved
            mode_dim=config.skill_dim,
            env=expl_env,
            sample_seqlen=config.seq_len,
        )

    elif variant == 'sampling_random_seq_len':
        expl_step_collector = SeqCollectorHorizonWholeSeqSaving(
            expl_env,
            policy,
            max_seqs=get_config_item(config, 'max_seqs', 5000),
            skill_selector=skill_selector,
        )
        replay_buffer = LatentReplayBufferSplitSeqSamplingRandomSeqLen(
            max_replay_buffer_size=config.replay_buffer_size,
            seq_len=config.horizon_len,  # Now whole horizon is saved
            mode_dim=config.skill_dim,
            env=expl_env,
            min_sample_seq_len=config.min_sample_seq_len,
            max_sample_seq_len=config.max_sample_seq_len,
        )

    else:
        raise NotImplementedError

    return replay_buffer, expl_step_collector
