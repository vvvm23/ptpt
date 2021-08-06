import torch
import torch.nn.functional as F

from pathlib import Path
from typing import List, Tuple, Callable
from functools import partial

from .utils import get_device

class TrainerConfig:
    exp_name:               str             = None,
    exp_dir:                str             = "exp",

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

class Trainer:
    def __init__(self,
        net:                    torch.nn.Module,
        loss_fn:                Callable,
        train_dataset:          torch.utils.data.Dataset,
        test_dataset:           torch.utils.data.Dataset,           
        device_fn:              Callable = lambda self, x: x.to(self.device),
        cfg:                    TrainerConfig = None
    ):
        if cfg == None:
            cfg = TrainerConfig()
        self.cfg = cfg

        self.device = get_device(cfg.use_cuda)
        self.net = net.to(self.device)

        self.opt = cfg.optimizer
        if not self.opt:
            self.opt = Trainer._get_opt(cfg)
        self.opt.zero_grad()

        self.lr_scheduler = None
        if cfg.lr_anneal_mode:
            self.lr_scheduler = Trainer._get_scheduler(cfg)

        self.grad_scaler = torch.cuda.amp.GradScaler(enabled = cfg.use_amp)

        self.nb_examples = 0
        self.nb_updates = 0

    # TODO: return correspond opt class (or instance maybe?) based on name
    def _get_opt(cfg):
        return torch.optim.Adam

    def _get_scheduler(cfg):
        pass

    def train(self):
        pass

    def train_step(self):
        pass
    def test_step(self):
        pass

    def _update_parameters(self):
        self.scaler.step(self.opt)
        self.opt.zero_grad()
        self.scaler.update()
        self.lr_scheduler.step()
        self.nb_examples = 0 # makes some assumptions about mini bs dividing bs perfectly
        self.nb_updates += 1

    # TODO: actually save to file at some internally calculated path
    def save_checkpoint(self):
        if not cfg.use_checkpoints:
            return

        checkpoint = {
            'net': self.net.state_dict(),
            'opt': self.opt.state_dict(),
            'scaler': self.scaler.state_dict(),
            'nb_examples': self.nb_examples,
            'nb_updates': self.nb_updates, 
        }

    # TODO: add init from load checkpoint option
    def load_checkpoint(self, path):
        checkpoint = torch.load(path)

        self.net.load_state_dict(checkpoint['net'])
        self.opt.load_state_dict(checkpoint['opt'])
        self.scaler.load_state_dict(checkpoint['scaler'])
        self.nb_examples = checkpoint['nb_examples']
        self.nb_updates = checkpoint['nb_updates']
