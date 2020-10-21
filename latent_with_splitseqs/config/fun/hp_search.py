from easydict import EasyDict as edict
import numpy as np
import copy

def get_random_hp(
        min_hp: edict,
        max_hp: edict,
):
    return_hp = copy.deepcopy(min_hp)

    _iterate_over_min_max_dicts(
        return_hp=return_hp,
        min_dict=min_hp,
        max_dict=max_hp,
    )

    return return_hp


def _iterate_over_min_max_dicts(return_hp, min_dict, max_dict, current_key_list=None):
    if current_key_list is None:
        current_key_list = []
    else:
        assert isinstance(current_key_list, list)

    for (key_min, hp_min), (key_max, hp_max) in zip(min_dict.items(), max_dict.items()):
        assert key_min == key_max
        assert type(hp_max) is type(hp_min)
        key = key_min

        if isinstance(hp_max, dict):
            # Recursion
            _iterate_over_min_max_dicts(
                return_hp=return_hp,
                min_dict=hp_min,
                max_dict=hp_max,
                current_key_list=current_key_list + [key],
            )

        else:
            hp_type = type(hp_max)
            if hp_min != hp_max:
                sampled_val = _sample_val(
                    min=hp_min,
                    max=hp_max,
                    hp_type=hp_type,
                )

                if key == 'batch_size':
                    pass

                _change_nested_dict_val(
                    key_list=current_key_list + [key],
                    nested=return_hp,
                    new_val=sampled_val,
                )




def _sample_val(hp_type, min, max):
    try:
        assert min < max
    except:
        raise ValueError
    if hp_type is int:
        sample = np.random.randint(
            low=min,
            high=max + 1,
        )

    elif hp_type is float:
        sample = np.random.uniform(
            low=min,
            high=max,
        )

    else:
        raise NotImplementedError

    assert min <= sample <= max
    return sample


def _change_nested_dict_val(key_list: list, new_val, nested: dict):
    cur = nested
    for idx, path_item in  enumerate(key_list[:-1]):
        try:
            cur = cur[path_item]
        except KeyError:
            assert idx == len(key_list) - 1
            assert path_item == key_list[-1]
            cur = cur[path_item] = {}

    cur[key_list[-1]] = new_val
