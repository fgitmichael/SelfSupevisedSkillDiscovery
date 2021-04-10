import numpy as np


class IterationInsertSorter:

    def __init__(self, _max_size,):
        self._max_size = _max_size
        self._size = 0
        self._data_array = np.array([])
        self._sort_idx = np.array([], dtype=np.int)
        self._sorted_data_array = np.array([])
        self._inv_sort_idx = np.array([], dtype=np.int)

        self.top = 0

    def add(self, data: np.ndarray, top: int):
        """
        Array data consiting of same values
        """
        assert self.top == top
        len_data = data.shape[0]
        full_with_new_data = self._size + len_data > self._max_size
        assert not full_with_new_data

        self._data_array = np.concatenate([self._data_array, data])
        insert_idx_ord = np.searchsorted(self._sorted_data_array, data[0])
        self._sorted_data_array = np.insert(self._sorted_data_array, insert_idx_ord, data)
        self._sort_idx = np.insert(self._sort_idx, insert_idx_ord, np.arange(top, top + len_data))
        self._inv_sort_idx = np.insert(self._inv_sort_idx, top, np.arange(insert_idx_ord, insert_idx_ord + len_data))
        self._inv_sort_idx[self._sort_idx[(insert_idx_ord + len_data):]] += len_data

        self.top += len_data
        assert np.all(self._data_array[self._sort_idx] == self._sorted_data_array)
        assert np.all(self._sorted_data_array[self._inv_sort_idx] == self._data_array)

    def delete(self, idx: np.ndarray):
        len_data = idx.shape[0]
        assert self.top - len_data >= 0

        self._data_array = np.delete(self._data_array, idx)
        self._sorted_data_array = np.delete(self._sorted_data_array, self._inv_sort_idx[idx])
        self._sort_idx = np.delete(self._sort_idx, self._inv_sort_idx[idx])
        self._inv_sort_idx = np.delete(self._inv_sort_idx, idx)

