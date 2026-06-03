import torch

def reshape_tensor(x, op):
    """
    Returns: list
    """
    x = torch.tensor(x).float()

    if op == "flatten":
        return x.flatten().tolist()
    elif op == "squeeze":
        return x.squeeze().tolist()
    elif op == "unsqueeze":
        return x.unsqueeze().tolist()
    elif op == "transpose":
        return x.permute(1, 0).tolist()
    
