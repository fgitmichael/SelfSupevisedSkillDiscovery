import numpy as np


def np_array_equality(*arrays):
    last_array = arrays[0]
    bool_var = False
    for array in arrays[1:]:
        bool_var = np.all(np.equal(last_array, array))
        if bool_var is False:
            break

    return bool_var
