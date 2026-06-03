import torch

def compute_gradient(values):
    """
    Returns: list of float gradient values dy/dx
    """
    values = [float(v) for v in values]
    x = torch.tensor(values, requires_grad=True)

    y = (x**3 + 2*x).sum()
    y.backward()

    return x.grad.tolist()
