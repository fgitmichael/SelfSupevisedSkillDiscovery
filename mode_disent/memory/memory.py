from collections import deque
import numpy as np
import torch

from code_slac.memory.lazy import LazyFrames


class MyLazySequenceBuff:
    keys = ['state', 'action', 'skill', 'done']

    def __init__(self, num_sequences):
        self.num_sequences = int(num_sequences)
        self.memory = {}

    def reset(self):
        self.memory = {}
        for key in self.keys[1:]:
            self.memory[key] = deque(maxlen=self.num_sequences)

        # There's always a next state
        self.memory['state'] = deque(maxlen=self.num_sequences + 1)

    def set_init_state(self, state):
        self.reset()
        self.memory['state'].append(state)

    def append(self, **kwargs):
        for key in self.keys:
            if key in kwargs:
                self.memory[key].append(kwargs[key])
            else:
                raise KeyError('Append Value is missing')

    def get(self):
        temp_dict = {}
        for key in self.keys:
            temp_dict[key] = LazyFrames(list(self.memory[key]))
        return temp_dict

    def __len__(self):
        return len(self.memory[self.keys[0]])


class MyLazyMemory(dict):
    keys = ('state', 'action', 'skill', 'done')

    def __init__(self,
                 state_rep,
                 capacity,
                 num_sequences,
                 observation_shape,
                 action_shape,
                 device):
        super(MyLazyMemory, self).__init__()
        if state_rep:
            dtypes = (np.float32, np.float32, np.uint8, np.bool)
        else:
            dtypes = (np.uint8, np.float32, np.uint8, np.bool)

        self.dtypes = {}
        for idx, key in enumerate(self.keys):
            self.dtypes[key] = dtypes[idx]

        self.capacity = int(capacity)
        self.num_sequences = int(num_sequences)
        self.observation_shape = observation_shape
        self.action_shape = action_shape
        self.device = device
        self.state_rep = state_rep
        self.is_set_init = False
        self._p = 0
        self._n = 0
        self.buff = None
        self.reset()

    def reset(self):
        self.is_set_init = False
        self._p = 0  # pointer for saving
        self._n = 0  # number of samples
        for key in self.keys:
            self[key] = [None] * self.capacity
        self.buff = MyLazySequenceBuff(num_sequences=self.num_sequences)
        assert self.keys == self.buff.keys

    def set_initial_state(self, state):
        self.buff.set_init_state(state)
        self.is_set_init = True

    def append(self, **kwargs):
        assert self.is_set_init is True

        self.buff.append(**kwargs)

        # State sequences length determines length of the buffer
        if len(self.buff) == self.num_sequences + 1:
            seq_dict = self.buff.get()
            self._append(seq_dict)

    def _append(self, seq_dict):
        for key in self.keys:
            if key in seq_dict:
                self[key][self._p] = seq_dict[key]
            else:
                raise KeyError('Part of Sequence is missing')

        self._n = min(self._n + 1, self.capacity)
        self._p = (self._p + 1) % self.capacity

    def sample_sequence(self, batch_size):
        """
        Return:
            states_seq   (N, S+1, *observation_shape) shaped tensor
            ...
        """
        indices = np.random.randint(low=0, high=self._n, size=batch_size)

        # TODO make this more generic
        states_seq = np.empty((
            batch_size, self.num_sequences+1, *self.observation_shape),
            dtype=self.dtypes['state'])
        actions_seq = np.empty(
            (batch_size, self.num_sequences, *self.action_shape),
            dtype=self.dtypes['action'])
        skill_seq = np.empty(
            (batch_size, self.num_sequences, 1),
            dtype=self.dtypes['skill'])
        dones_seq = np.empty(
            (batch_size, self.num_sequences, 1),
            dtype=self.dtypes['done']
        )

        for i, idx in enumerate(indices):
            # Convert LazyFrames to np.ndarray
            states_seq[i, :, :] = self['state'][idx]
            actions_seq[i, :, :] = self['action'][idx]
            skill_seq[i, :, :] = self['skill'][idx]
            dones_seq[i, :, :] = self['dones'][idx]

        if self.state_rep:
            states_seq = torch.from_numpy(states_seq).float().to(self.device)
        else:
            states_seq = torch.from_numpy(states_seq).int().to(self.device)

        actions_seq = torch.from_numpy(actions_seq).float().to(self.device)
        skill_seq = torch.from_numpy(skill_seq).int().to(self.device)
        dones_seq = torch.from_numpy(dones_seq).bool().to(self.device)

        return {'states_seq': states_seq,
                'actions_seq': actions_seq,
                'skill_seq': skill_seq,
                'dones_seq': dones_seq}
