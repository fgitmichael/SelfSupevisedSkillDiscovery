from typing import Union

from my_utils.dicts.dict_keylist import get_item_using_keylist, keylist_in_dict


def get_config_item(
        config: dict,
        key: Union[str, list, tuple],
        default=None
):
    if isinstance(key, str):
        if key in config.keys():
            ret_val = config[key]
        else:
            ret_val = default

    elif isinstance(key, list) or isinstance(key, tuple):
        if keylist_in_dict(config, key):
            ret_val = get_item_using_keylist(config, key)
        else:
            ret_val = default

    else:
        raise ValueError

    return ret_val


