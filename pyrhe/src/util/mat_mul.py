import torch
import numpy as np
    
def to_tensor(x, device=None):
    """
    Convert a NumPy array to a PyTorch tensor.
    """
    if isinstance(x, torch.Tensor):
        return x.to(device=device)
    
    elif isinstance(x, np.ndarray):
        return torch.from_numpy(x).float().to(device=device)
    
    else:
        raise ValueError(f"Failed to convert {x} to tensor: {x} is neither a tensor or a Numpy array. ")

def mat_mul(*mats, device, to_numpy=True):
    """
    Perform matrix multiplication on multiple inputs.
    All inputs are first converted to tensors on the given device.
    """
    if not mats:
        raise ValueError("At least one matrix is required.")
    result_tensor = to_tensor(mats[0], device)
    for mat in mats[1:]:
        tensor = to_tensor(mat, device)
        result_tensor = result_tensor @ tensor
    if to_numpy:
        return result_tensor.cpu().numpy()
    else:
        return result_tensor


def elem_mul(*mats, device, to_numpy=True):
    """
    Perform element-wise multiplication on multiple inputs.
    All inputs are first converted to tensors on the given device.
    """
    if not mats:
        raise ValueError("At least one matrix is required.")
    result_tensor = to_tensor(mats[0], device)
    for mat in mats[1:]:
        tensor = to_tensor(mat, device)
        result_tensor = result_tensor * tensor
    if to_numpy:
        return result_tensor.cpu().numpy()
    else:
        return result_tensor