import abc

import self_supervised.utils.typed_dicts as td


class RollouterBase(object):

    @abc.abstractmethod
    def do_rollout(
            self,
            seq_len) -> td.TransitionMapping:
        pass

    @abc.abstractmethod
    def reset(self):
        pass
