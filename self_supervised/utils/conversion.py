import rlkit.torch.pytorch_util as ptu


def np_dict_to_torch(np_dict):
    result = dict()
    for k, v in np_dict:
        result[k] = ptu.from_numpy(v)

    return result


