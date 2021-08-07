import torch
import torch.nn.functional as F

import datetime
from pathlib import Path
from typing import List, Tuple, Callable
from functools import partial

from .utils import get_device
from .log import debug, info, warning, error, critical

class TrainerConfig:
    """ 
    Helper class to store Trainer configuration options and some sane
    defaults.

    Attributes:
        exp_name:               identifying name of the current experiment; used
                                for creating experiment directory.

        exp_dir:                root directory for all experiments.

        batch_size:             number of training examples before updating
                                parameters.

        mini_batch_size:        number of examples on the GPU at any one time.

        nb_batches:             tuple (nb_train, nb_test) representing number of
                                train batches before evaluating, then how many
                                eval batches before resuming training.
                                a value of 0 means to consume entirety of loader
                                before continuing to next stage.

        max_steps:              maximum number of steps before termination.
                                a value of 0 means there is no predefined
                                maximum number of steps.

        optimizer:              a `torch.optim` instance, if present use this
                                optimizer over `optimizer_name`.

        optimizer_name:         a string that maps to some optimizer so it can
                                be automatically initialised.

        learning_rate:          the optimizer learning rate.

        lr_scheduler_name:      string representing the learning rate scheduler
                                name. if present, create the corresponding rate
                                scheduler.

        lr_milestones:          if we are using a scheduler, define a list of
                                milestones, based on `nb_updates`.

        lr_gamma:               if the scheduler uses gamma annealing, define
                                the multiplicative factor.

        nb_workers:             number of CPU workers to use when loading data.

        use_cuda:               whether to use the CUDA device if available

        use_amp:                whether to try using automatic mixed precision.

        save_outputs:           whether to save checkpoints, logs and other data
                                to disk.

        checkpoint_frequency:   the frequency at which to save a checkpoint,
                                measured in `nb_updates`.

        metric_names:           list of metric names returned by `loss_fn`.
                                this can be empty, in the case that no additional
                                metrics (other than the loss itself) are returned.
    """

    exp_name:               str             = "exp"
    exp_dir:                str             = "exp"

    batch_size:             int             = 1
    mini_batch_size:        int             = 1
    nb_batches:             Tuple[int]      = (0, 0)
    max_steps:              int             = 0

    optimizer:              torch.optim     = None
    optimizer_name:         str             = 'adam'

    learning_rate:          float           = 1e-4
    lr_scheduler_name:      str             = None
    lr_milestones:          List[int]       = None
    lr_gamma:               float           = None

    nb_workers:             int             = 0
    use_cuda:               bool            = True
    use_amp:                bool            = True

    save_outputs:           bool            = True
    checkpoint_frequency:   int             = 1000

    metric_names:           List[str]       = []

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)

    def __str__(self):
        attributes = [x for x in dir(self) if not x.startswith('__')]
        return '\n'.join(f"{a:25}: {getattr(self, a)}" for a in attributes)

class Trainer:
    """
    Module core class that abstracts away PyTorch implementation details. 

    The aim is the provide the following:
        - abstracting away repeated details common across many deep learning
          projects.
        - providing a suite of deep learning methods that can be easily toggled
          and changed.
        - abstract away complex implementation details such as multi-GPU, TPUs
          and AMP; both in a local and distributed setting.
        - providing a consistent and ever-present API for logging and
          experiment reproducibility.

    Essentially, to abstract away as much as possible, whilst maintaining
    flexibility, and to make using best practises as painless as possible.
    """
    def __init__(self,
        net:                    torch.nn.Module,
        loss_fn:                Callable,
        train_dataset:          torch.utils.data.Dataset,
        test_dataset:           torch.utils.data.Dataset,           
        device_fn:              Callable = lambda self, x: x.to(self.device),
        cfg:                    TrainerConfig = None
    ):
        """
        the `Trainer` init function.

        Args:
            net:            a `nn.Module` that is the model we wish to train.
            loss_fn:        the loss function we wish to minimise that calls
                            `self.net`.
            train_dataset:  the training dataset.
            test_dataset:   the test dataset.
            device_fn:      a function that handles moving a batch to
                            `self.device`.
            cfg:            a `TrainerConfig` instance that holds all
                            hyperparameters.

        TODO: `loss_fn` and `device_fn` having `self` in arg list feels awkward.
        find alternative.
            - turns out, they do not need this at all. `loss_fn` can access `net` 
              even after passing it to the class.
            - device function is less clear cut.
        """
        if cfg == None:
            info("no TrainerConfig specified. assuming default options.")
            cfg = TrainerConfig()
        self.cfg = cfg

        if self.cfg.save_outputs:
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

        self._loss_fn = loss_fn
        self.loss_fn = self._autocast_loss if cfg.use_amp else self._loss_fn
        if cfg.use_amp:
            info("using automatic mixed precision")

        self.device_fn = partial(device_fn, self)

        self.nb_examples = 0
        self.nb_updates = 0

    def _get_opt(self):
        """
        get the optimizer based on `cfg.optimizer_name`
        defaults to the Adam optimizer.
        """
        if cfg.optimizer_name in ['adam']:
            info("using Adam optimizer")
            return torch.optim.Adam(self.net.parameters(), lr=self.cfg.learning_rate)

        if cfg.optimizer_name is not None:
            warning("unrecognised optimizer name. defaulting to 'adam'")
        info("using Adam optimizer")
        return torch.optim.Adam(self.net.parameters(), lr=self.cfg.learning_rate)

    def _get_scheduler(self):
        """
        gets the learning rate scheduling mode.
        defaults to no scheduling, i.e: the identity scheduler.
        """
        if cfg.lr_scheduler_name in ['multi', 'multisteplr']:
            info("using MultiStepLR learning rate scheduler")
            return torch.optim.lr_scheduler.MultiStepLR(
                self.opt, 
                milestones = self.cfg.lr_milestones, 
                gamma = self.cfg.lr_gamma,
            )
        
        if cfg.lr_scheduler_name is not None:
            warning("unrecognised annealing mode. defaulting to no lr scheduler.")
        return torch.optim.lr_scheduler.MultiStepLR(
            self.opt,
            milestones = [],
            gamma = 1.0,
        )

    def _setup_workspace(self):
        """
        function that sets up the workspace directories and stores them in
        `self.directories`.

        TODO: add support for additional output directories (think: images)
        TODO: might be good to add support for arbitrary callback functions 
        """
        info("setting up experiment workspace")
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
        info(f"experimental directory is: {self.directories['root']}")
        info("done setting up directories")

    def _setup_dataloader(self, train_dataset, test_dataset):
        """
        sets up dataloaders for provided datasets.
        also transparently converts finite datasets to infinite ones by setting
        `self.nb_batches` equal to length of dataloader.
        """
        info("setting up dataloaders")
        args = {
            'batch_size': self.cfg.mini_batch_size,
            'shuffle': True,
            'num_workers': self.cfg.nb_workers,
        }

        self.train_loader = torch.utils.data.DataLoader(train_dataset, **args)
        self.test_loader = torch.utils.data.DataLoader(test_dataset, **args)

        self.train_iter = iter(self.train_loader)
        self.test_iter = iter(self.test_loader)

        self.nb_batches = (
            self.cfg.nb_batches[0] if self.cfg.nb_batches[0] else len(self.train_loader),
            self.cfg.nb_batches[1] if self.cfg.nb_batches[1] else len(self.test_loader),
        )
        info("done setting up dataloaders")

    def _get_batch(self, split='train'):
        """
        gets a batch of data from the specified split.
        if the iterator has been exhausted, create a new one from the loader.

        TODO: load data onto device based on device_fn
        TODO: eventually get rid of device_fn, and automatically determine based
        on specified device mode (single vs. multi device / process, CPU vs.
        GPU vs. TPU)
        """
        if split == 'train':
            iterator = self.train_iter
            loader = self.train_loader
        elif split in ['test', 'eval']:
            iterator = self.test_iter
            loader = self.test_loader

        try:
            data = next(iterator)
        except StopIteration:
            debug(f"StopIteration - refreshing dataloader for split '{split}'")
            iterator = iter(loader)
            data = next(iterator)
            if split == 'train':
                self.train_iter = iterator
            elif split in ['test', 'eval']:
                self.test_iter = iterator

        return data

    def _autocast_loss(self, *args):
        """
        thin wrapper around `loss_fn` to provide AMP autocasting
        """
        with torch.cuda.amp.autocast(enabled=self.scaler.is_enabled()):
            return self._loss_fn(*args)

    def _check_terminate(self) -> bool:
        """
        function that return `True` if a training termination condition has
        occurred.

        TODO: add some common termination conditions
        TODO: add option to pass arbitrary termination conditions
        """
        if self.cfg.max_steps and self.nb_updates < self.cfg.max_steps:
            return True

        return False

    def train(self, 
            tqdm = False, 
            silent = False,
        ):
        """
        starts the main training loop. continue until termination condition is
        met.

        Args:
            tqdm: train with TQDM loading bars
            silent: run training loop silently

        TODO: a lot
        """
        while not self._check_terminate():
            for _ in self.nb_batches[0]:
                loss, *metrics = train_step()

            for _ in self.nb_batches[1]:
                loss, *metrics = eval_step()

    def train_step(self):
        """
        executes one iteration of the training loop.
        one iteration looks like:
            - getting a batch
            - calculating the loss (and other metrics)
            - weight loss based on how much of the batch was complete
            - if one full bach has been processed, update the parameters

        essentially, transparently implements 'gradient accumulation'; helpful
        for super-massive batch sizes.

        TODO: issue if mini bs doesn't divide bs perfectly
        TODO: need way to understand meaning of metrics
        """
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
        """
        executes one evaluation step.
        very similar to `train_step`, but only calculates metrics and returns
        them and does not calculate gradients.
        """
        self.net.eval()
        batch = self._get_batch(split='eval')
        loss, *metrics = self.loss_fn(batch)
        return loss, metrics

    def _update_parameters(self):
        """
        updates the parameters in `self.net` based on accumulated gradients.
        also updates schedulers, scalers and other variables.
        """
        self.scaler.step(self.opt)
        self.opt.zero_grad()
        self.scaler.update()
        self.lr_scheduler.step()
        self.nb_examples = 0 # makes some assumptions about mini bs dividing bs perfectly
        self.nb_updates += 1

    def save_checkpoint(self):
        """
        saves a checkpoint to `self.directories['checkpoints']`
        returns early if `cfg` specifies not to save outputs
        """
        if not cfg.save_outputs:
            return
        checkpoint_name = f"checkpoint-{str(self.nb_updates).zfill(7)}.pt"
        info(f"saving checkpoint '{checkpoint_name}'")

        checkpoint = {
            'net': self.net.state_dict(),
            'opt': self.opt.state_dict(),
            'scaler': self.scaler.state_dict(),
            'nb_examples': self.nb_examples,
            'nb_updates': self.nb_updates, 
        }
        torch.save(checkpoint, self.directories['checkpoints'] / checkpoint_name)

    def load_checkpoint(self, path):
        """
        restore `Trainer` using checkpoint at path specified in `path`.

        TODO: add init from load checkpoint option
        """
        info(f"restoring from checkpoint '{path}'")
        checkpoint = torch.load(path)

        self.net.load_state_dict(checkpoint['net'])
        self.opt.load_state_dict(checkpoint['opt'])
        self.scaler.load_state_dict(checkpoint['scaler'])
        self.nb_examples = checkpoint['nb_examples']
        self.nb_updates = checkpoint['nb_updates']