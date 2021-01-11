from typing import Union


def get_item_using_keylist(
        dict_: dict,
        keylist: Union[list, tuple],
):
    el = dict_
    for key in keylist:
        try:
            el = el[key]
        except:
            KeyError("Keys of keylist are present in the dictionary")
    return el


def keylist_in_dict(
        dict_: dict,
        keylist: Union[list, tuple],
) -> bool:
    key_in_dictkeys = True
    el = dict_
    for key in keylist:
        if key in el:
            el = el[key]
        else:
            key_in_dictkeys = False
            break

    return key_in_dictkeys


def _check_keylist(keylist):
    assert isinstance(keylist, list) or isinstance(keylist, tuple)

