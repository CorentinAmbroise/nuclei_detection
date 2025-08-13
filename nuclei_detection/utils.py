import torch


def get_device():
    """
    Returns the device to be used for PyTorch operations.
    If a GPU is available, it returns 'cuda', otherwise it returns 'cpu'.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    return device
