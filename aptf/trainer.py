import torch
from typing import List, Tuple

class TrainerConfig:
    batch_size:             int             = 1
    nb_batches:             Tuple[int]      = (0, 0)
    max_steps:              int             = 0

    optimizer:              torch.optim     = None
    optimizer_name:         str             = 'adam'

    learning_rate:          float           = 1e-4
    lr_anneal_mode:         str             = None
    lr_milestones:          List[int]       = None

    nb_workers:             int             = 0
    use_cuda:               bool            = True
    use_amp:                bool            = True

    use_checkpoints:        bool            = True
    checkpoint_frequency:   int             = 100

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)
