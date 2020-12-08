import copy


def remove_nones(in_dict: dict) -> dict:
    out_dict = {}
    for k, v in in_dict.items():
        if v is not None:
            out_dict[k] = v

    return out_dict