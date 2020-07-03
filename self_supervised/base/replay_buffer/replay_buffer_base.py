import abc

from rlkit.data_management.replay_buffer import ReplayBuffer

from self_supervised.base.data_collector.rollout import PathMapping


# Adding skills
class SequenceReplayBuffer(ReplayBuffer, metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def add_sample(self,
                   observation,
                   action,
                   reward,
                   next_observation,
                   terminal,
                   **kwargs):
        pass

    @abc.abstractmethod
    def terminate_episode(self):
        """
        Let the replay buffer know that the episode has terminated in case some
        special book-keeping has to happen.
        :return:
        """
        pass

    @abc.abstractmethod
    def num_steps_can_sample(self, **kwargs):
        """
        :return: # of unique items that can be sampled.
        """
        pass

    def add_path(self, path: PathMapping):
        """
        Add a path to the replay buffer.

        This default implementation naively goes through every step, but you
        may want to optimize this.

        NOTE: You should NOT call "terminate_episode" after calling add_path.
        It's assumed that this function handles the episode termination.

        :param path: Dict like one outputted by rlkit.samplers.util.rollout
        """
        for i, (
                obs,
                action,
                reward,
                next_obs,
                terminal,
                agent_info,
                env_info,
                skills
        ) in enumerate(zip(
            path.observations,
            path.actions,
            path.rewards,
            path.next_observations,
            path.terminals,
            path.agent_infos,
            path.env_infos,
            path.skills
        )):
            self.add_sample(
                observation=obs,
                action=action,
                reward=reward,
                next_observation=next_obs,
                terminal=terminal,
                agent_info=agent_info,
                env_info=env_info,
                skill=skills
            )
        self.terminate_episode()

    def add_paths(self, paths):
        for path in paths:
            self.add_path(path)

    @abc.abstractmethod
    def random_batch(self, batch_size):
        """
        Return a batch of size `batch_size`.
        :param batch_size:
        :return:
        """
        pass

    def get_diagnostics(self):
        return {}

    def get_snapshot(self):
        return {}

    def end_epoch(self, epoch):
        return

