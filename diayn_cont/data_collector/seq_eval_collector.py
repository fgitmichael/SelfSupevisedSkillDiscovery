from collections import deque, OrderedDict

from rlkit.samplers.data_collector.path_collector import MdpPathCollector


class MdpPathCollectorWithReset(MdpPathCollector):
    def reset(self):
        self._epoch_paths = deque(maxlen=self._max_num_epoch_paths_saved)
