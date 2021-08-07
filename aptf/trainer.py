import torch
import torch.nn.functional as F

import datetime
from pathlib import Path
from typing import List, Tuple, Callable
from functools import partial

from .utils import get_device

class TrainerConfig:
    exp_name:               str             = "exp",
    exp_dir:                str             = "exp",

    batch_size:             int             = 1
    mini_batch_size:        int             = 1
    nb_batches:             Tuple[int]      = (0, 0)
    max_steps:              int             = 0

    optimizer:              torch.optim     = None
    optimizer_name:         str             = 'adam'

    learning_rate:          float           = 1e-4
    lr_anneal_mode:         str             = None
    lr_milestones:          List[int]       = None
    lr_gamma:               float           = None

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
        self._setup_workspace()
        self._setup_dataloader()

        self.device = get_device(cfg.use_cuda)
        self.net = net.to(self.device)

        self.opt = cfg.optimizer
        if not self.opt:
            self.opt = self._get_opt()
        self.opt.zero_grad()

        self.lr_scheduler = self._get_scheduler()

        self.grad_scaler = torch.cuda.amp.GradScaler(enabled = cfg.use_amp)

        self._loss_fn = partial(loss_fn, self)
        self.loss_fn = self._autocast_loss if cfg.use_amp else self._loss_fn

        self.device_fn = partial(device_fn, self)

        self.nb_examples = 0
        self.nb_updates = 0

    def _get_opt(self):
        if cfg.optimizer_name in ['adam']:
            return torch.optim.Adam(self.net.parameters(), lr=self.cfg.learning_rate)

        if cfg.optimizer_name is not None:
            print("warning: unrecognised optimizer name. defaulting to 'adam'")
        return torch.optim.Adam(self.net.parameters(), lr=self.cfg.learning_rate)

    def _get_scheduler(self):
        if cfg.lr_anneal_mode in ['multi', 'multisteplr']
            return torch.optim.lr_scheduler.MultiStepLR(
                self.opt, 
                milestones = self.cfg.lr_milestones, 
                gamma = self.cfg.lr_gamma,
            )
        
        if cfg.lr_anneal_mode is not None:
            print("warning: unrecognised annealing mode. defaulting to no lr scheduler.")
        return torch.optim.lr_scheduler.MultiStepLR(
            self.opt,
            milestones = [],
            gamma = 1.0,
        )

    def _setup_workspace(self):
        exps_dir = Path(cfg.exp_dir)
        exps_dir.mkdir(exist_ok=True)

        self.save_id = (
            cfg.exp_name +
            str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        )
        exp_root = exps_dir / self.save_id
        exp_root.mkdir(exist_ok=True)

        checkpoint_dir = exp_root / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)

        log_dir = exp_root / "logs"
        log_dir.mkdir(exist_ok=True)

        self.directories = {
            'root': exp_root,
            'checkpoints': checkpoint_dir,
            'logs': log_dir,
        }

    def _setup_dataloader(self, train_dataset, test_dataset):
        args = {
            batch_size = self.cfg.mini_batch_size,
            shuffle = True,
            num_workers = self.cfg.nb_workers,
        }

        self.train_loader = torch.utils.data.DataLoader(train_dataset, **args)
        self.test_loader = torch.utils.data.DataLoader(test_dataset, **args)

        self.train_iter = iter(self.train_loader)
        self.test_iter = iter(self.test_loader)

        self.nb_batches = (
            self.cfg.nb_batches[0] if self.cfg.nb_batches[0] else len(self.train_loader),
            self.cfg.nb_batches[1] if self.cfg.nb_batches[1] else len(self.test_loader),
        )

    def _get_batch(self, split='train'):
        if split == 'train':
            iterator = self.train_iter
            loader = self.train_loader
        elif split in ['test', 'eval']:
            iterator = self.test_iter
            loader = self.test_loader

        try:
            data = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            data = next(iterator)
            if split == 'train':
                self.train_iter = iterator
            elif split in ['test', 'eval']:
                self.test_iter = iterator

        return data

    def _autocast_loss(self, *args):
        with torch.cuda.amp.autocast(enabled=self.scaler.is_enabled()):
            return self._loss_fn(*args)

    def _check_terminate(self):
        if self.cfg.max_steps < self.nb_updates:
            return True

        return False

    def train(self, 
            tqdm = False, 
            silent = False
        ):
        while not self._check_terminate():
            for _ in self.nb_batches[0]:
                loss, *metrics = train_step()

            for _ in self.nb_batches[1]:
                loss, *metrics = eval_step()

    # TODO: issue if mini bs doesn't divide bs correctly
    def train_step(self):
        self.net.train()
        batch = self._get_batch(split='train')
        loss, *metrics = self.loss_fn(batch)

        if isinstance(batch, (list, tuple)):
            mini_batch_size = batch[0].shape[0]
        else:
            mini_batch_size = batch.shape[0]

        batch_weighting = mini_batch_size / self.cfg.batch_size
        self.scaler.scale(loss * batch_weighting).backward()

        self.nb_examples += mini_batch_size
        if self.nb_examples >= self.cfg.batch_size:
            self._update_parameters()
        return loss, metrics

    @torch.no_grad()
    def eval_step(self):
        self.net.eval()
        batch = self._get_batch(split='eval')
        loss, *metrics = self.loss_fn(batch)
        return loss, metrics

    def _update_parameters(self):
        self.scaler.step(self.opt)
        self.opt.zero_grad()
        self.scaler.update()
        self.lr_scheduler.step()
        self.nb_examples = 0 # makes some assumptions about mini bs dividing bs perfectly
        self.nb_updates += 1

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
        torch.save(checkpoint, self.directories['checkpoints'] / f"checkpoint-{str(self.nb_updates).zfill(7)}.pt")

    # TODO: add init from load checkpoint option
    def load_checkpoint(self, path):
        checkpoint = torch.load(path)

        self.net.load_state_dict(checkpoint['net'])
        self.opt.load_state_dict(checkpoint['opt'])
        self.scaler.load_state_dict(checkpoint['scaler'])
        self.nb_examples = checkpoint['nb_examples']
        self.nb_updates = checkpoint['nb_updates']
