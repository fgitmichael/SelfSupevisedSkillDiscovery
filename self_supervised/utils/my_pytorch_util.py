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


def eval(module: torch.nn.Module, *args, **kwargs):
    train_mode_before = module.training
    module.eval()
    with torch.no_grad():
        ret_val = module(*args, **kwargs)
    module.train(train_mode_before)
    return ret_val


def set_gpu_mode(mode: bool):
    if mode:
        assert torch.cuda.is_available()
    ptu.set_gpu_mode(mode)
