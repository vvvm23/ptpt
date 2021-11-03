import torch
import numpy as np
import random

class HelperModule(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.build(*args, **kwargs)

    def build(self, *args, **kwargs):
        raise NotImplementedError

def get_parameter_count(net: torch.nn.Module) -> int:
    return sum(p.numel() for p in net.parameters() if p.requires_grad)

# TODO: expand to support multi-gpu, distributed and TPU
def get_device(try_cuda):
    if not try_cuda or not torch.cuda.is_available(): return torch.device('cpu')
    return torch.device('cuda')

def set_seed(seed = None, min_seed=0, max_seed=2**8):
    if seed == None:
        seed = random.randint(min_seed, max_seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.random.manual_seed(seed)
    return seed
