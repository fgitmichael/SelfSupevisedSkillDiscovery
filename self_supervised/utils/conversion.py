from typing import Union
import self_supervised.utils.typed_dicts as td
import numpy as np
import sys

import rlkit.torch.pytorch_util as ptu


def from_numpy(
        numpy_obj: Union[td.TransitionMapping, td.TransitionModeMappingTorch])\
        -> dict:
#        -> Union[td.TransitionMapping, td.TransitionModeMappingTorch]:
    #class_name = numpy_obj.__class__.__name__
    #constr = getattr(sys.modules[__name__], class_name)

    #return constr(
    #    {
    #        k: ptu.from_numpy(v[k]) for k, v in numpy_obj
    #    }
    #)

    #return {
    #    k: (ptu.from_numpy(v) if type(v) is np.ndarray else v) \
    #    for k, v in numpy_obj.items()
    #}

    ret_dict = {}
    for k, v in numpy_obj.items():
        ret_dict[k] = ptu.from_numpy(v) if type(v) is np.ndarray else v

    return ret_dict

