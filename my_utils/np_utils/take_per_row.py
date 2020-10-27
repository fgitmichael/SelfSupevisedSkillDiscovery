# Source
# https://stackoverflow.com/questions/46091111/
# python-slice-array-at-different-position-on-every-row
import numpy as np


def take_per_row(A, indx, num_elem=2):
    all_indx = indx[:,None] + np.arange(num_elem)
    return A[np.arange(all_indx.shape[0])[:,None], all_indx]