from typing import Union
from self_supervised.utils.typed_dicts import *
import sys

import rlkit.torch.pytorch_util as ptu


def from_numpy(
        numpy_obj: Union[TransitionMapping, TransitionModeMappingTorch])\
        -> Union[TransitionMapping, TransitionModeMappingTorch]:
    class_name = numpy_obj.__class__.__name__
    constr = getattr(sys.modules[__name__], class_name)

    return constr(
        {
            k: ptu.from_numpy(v[k]) for k, v in numpy_obj
        }
    )
