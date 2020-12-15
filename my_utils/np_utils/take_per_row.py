# Source
# https://stackoverflow.com/questions/46091111/
# python-slice-array-at-different-position-on-every-row
import numpy as np


def take_per_row(A, indx, num_elem=2):
    indx = np.expand_dims(indx, axis=1)
    all_indx = indx + np.arange(num_elem)
    first_dim_indx = np.expand_dims(
        np.arange(all_indx.shape[0]),
        axis=1
    )
    return A[first_dim_indx, all_indx]
