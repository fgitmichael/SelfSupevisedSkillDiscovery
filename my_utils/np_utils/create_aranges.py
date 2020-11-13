import numpy as np

def create_aranges(
        stop,
        start: np.ndarray = None,
        step: np.ndarray = None,
        dtype = None,
):
    """
    Returns matrix with element aranges for the start, stop, step arrays

    Args:
        start           : (N,) numpy array
        stop            : (N,) numpy array
        step            : (N,) numpy array
        dtype           : numpy dtype

    """
    if start is None:
        start = np.zeros_like(stop)

    if step is None:
        step = 1

    assert start.shape == stop.shape
    assert np.all(
        (stop - start) == (stop - start)[0]
    )
    assert type(step) is int

    length = (stop - start)[0]
    return np.expand_dims(start, axis=-1) + np.arange(length, step=step, dtype=dtype)
