import torch
    
def to_tensor(mat, device):
    print("*********", device)
    return torch.from_numpy(mat).float().to(device)

def mat_mul(*mats, device, to_numpy=True):
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