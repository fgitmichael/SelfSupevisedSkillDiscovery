import torch
import rlkit.torch.pytorch_util as ptu

def rand(*args, torch_device=None, **kwargs):
    if torch_device is None:
        torch_device = ptu.device
    return torch.rand(*args, **kwargs, device=torch_device)

def eye(*args, torch_device=None, **kwargs):
    if torch_device is None:
        torch_device = ptu.device
    return torch.eye(*args, **kwargs, device=torch_device)

def get_oh_float(num_classes: int, labels):
    """
    Args:
        num_classes             : number of classes
        labels                  : list or 1dim array or 1dim tensor
    """
    return eye(num_classes)[labels]

def tensor_equality(*tensors):
    last_tensor = tensors[0]
    bool_var = False
    for tensor in tensors[1:]:
        bool_var = torch.all(
            torch.eq(
                last_tensor,
                tensor
            )
        )
        if bool_var is False:
            break

    return bool_var

@torch.no_grad()
def eval(module: torch.nn.Module, *args, **kwargs):
    train_mode_before = module.training
    module.train(False)
    try:
        ret_val = module(*args, **kwargs)
    except:
        raise ValueError
    module.train(train_mode_before)
    return ret_val
