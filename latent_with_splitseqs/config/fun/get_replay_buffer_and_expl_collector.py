from typing import Tuple

from self_supervised.base.replay_buffer.replay_buffer_base import SequenceReplayBuffer

from latent_with_splitseqs.memory.replay_buffer_for_latent import LatentReplayBuffer
from latent_with_splitseqs.base.replay_buffer_latent_splitseq_sampling_base \
    import LatentReplayBufferSplitSeqSamplingBase
from latent_with_splitseqs.memory.replay_buffer_latent_splitseq_sampling_fixed_seqlen \
    import get_fixed_seqlen_latent_replay_buffer_class
from latent_with_splitseqs.memory.replay_buffer_latent_splitseq_sampling_random_seqlen \
    import get_random_seqlen_latent_replay_buffer_class
from latent_with_splitseqs.data_collector.seq_collector_over_horizon_splitseq_save \
    import SeqCollectorHorizonSplitSeqSaving
from latent_with_splitseqs.data_collector.seq_collector_over_horizon_wholeseq_save \
    import SeqCollectorHorizonWholeSeqSaving
from latent_with_splitseqs.base.\
    replay_buffer_latent_splitseq_sampling_base_terminal_handling_memory_efficient \
    import LatentReplayBufferSplitSeqSamplingBaseMemoryEfficient

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
    terminal_handling = get_config_item(
        config=config,
        key='terminal_handling',
        default=False,
    )
    terminal_handling_save_memory = get_config_item(
        config=config,
        key='terminal_handling_save_memory',
        default=False,
    )
    min_sample_seqlen = get_config_item(
        config=config,
        key='min_sample_seqlen',
        default=2,
    )

    if variant == 'fixed':
        expl_step_collector = SeqCollectorHorizonSplitSeqSaving(
            expl_env,
            policy,
            max_seqs=get_config_item(config, 'max_seqs', 5000),
            skill_selector=skill_selector,
            terminal_handling=terminal_handling,
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
            terminal_handling=terminal_handling,
        )
        #replay_buffer = LatentReplayBufferSplitSeqSamplingFixedSeqLen(
        #    max_replay_buffer_size=config.replay_buffer_size,
        #    seq_len=config.horizon_len, # Now whole horizon is saved
        #    mode_dim=config.skill_dim,
        #    env=expl_env,
        #    sample_seqlen=config.seq_len,
        #    min_sample_seqlen=min_sample_seqlen,
        #)

        padding = get_config_item(
            config=config,
            key='padding',
            default=True,
        )

        if terminal_handling:
            base_replay_buffer_class = \
                LatentReplayBufferSplitSeqSamplingBaseMemoryEfficient
        else:
            base_replay_buffer_class = LatentReplayBufferSplitSeqSamplingBase

        replay_buffer_cls = get_fixed_seqlen_latent_replay_buffer_class(
            base_replay_buffer_class,
        )
        replay_buffer = replay_buffer_cls(
            max_replay_buffer_size=config.replay_buffer_size,
            seq_len=config.horizon_len,  # Now whole horizon is saved
            mode_dim=config.skill_dim,
            env=expl_env,
            sample_seqlen=config.seq_len,
            min_sample_seqlen=min_sample_seqlen,
            padding=padding,
        )

    elif variant == 'sampling_random_seq_len':
        expl_step_collector = SeqCollectorHorizonWholeSeqSaving(
            expl_env,
            policy,
            max_seqs=get_config_item(config, 'max_seqs', 5000),
            skill_selector=skill_selector,
            terminal_handling=terminal_handling,
        )
        #replay_buffer = LatentReplayBufferSplitSeqSamplingRandomSeqLen(
        #    max_replay_buffer_size=config.replay_buffer_size,
        #    seq_len=config.horizon_len,  # Now whole horizon is saved
        #    mode_dim=config.skill_dim,
        #    env=expl_env,
        #    min_sample_seqlen=min_sample_seqlen,
        #    min_sample_seq_len=config.min_sample_seq_len,
        #    max_sample_seq_len=config.max_sample_seq_len,
        #)

        padding = get_config_item(
            config=config,
            key='padding',
            default=True,
        )

        replay_buffer_cls = get_random_seqlen_latent_replay_buffer_class(
            LatentReplayBufferSplitSeqSamplingBase,
        )
        replay_buffer = replay_buffer_cls(
            max_replay_buffer_size=config.replay_buffer_size,
            seq_len=config.horizon_len,  # Now whole horizon is saved
            mode_dim=config.skill_dim,
            env=expl_env,
            min_sample_seqlen=min_sample_seqlen,
            min_sample_seq_len=config.min_sample_seq_len,
            max_sample_seq_len=config.max_sample_seq_len,
            padding=padding,
        )

    else:
        raise NotImplementedError

    return replay_buffer, expl_step_collector
