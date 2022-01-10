# Based on HuggingFace Transformers implementation of getting learning rate
# schedulers.
# -> https://github.com/huggingface/transformers/blob/v4.15.0/src/transformers/optimization.py

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

import math
from enum import Enum
from typing import Union, Optional, Dict

from .log import error

class SchedulerType(Enum):
    Constant = "constant"
    ConstantWithWarmup = "constant_with_warmup"
    Linear = "linear"
    LinearWithWarmup = "linear_with_warmup"
    Cosine = "cosine"
    CosineWithWarmup = "cosine_with_warmup"
    CosineWithRestartsAndWarmup = "cosine_with_restarts_with_warmup"
    Polynomial = "polynomial"
    PolynomialWithWarmup = "polynomial_with_warmup"

def get_constant_schedule(
        optimizer: Optimizer,
        last_epoch: int = -1,
    ):
    return LambdaLR(optimizer, lambda _: 1, last_epoch=last_epoch)

def get_constant_schedule_with_warmup(
        optimizer: Optimizer, 
        nb_warmup_steps: int, 
        last_epoch: int = -1,
    ):
    return LambdaLR(optimizer, lambda i: (
        float(i) / float(max(1.0, nb_warmup_steps)) if i < nb_warmup_steps else 1.0
    ), last_epoch=last_epoch)

def get_linear_schedule(
        optimizer: Optimizer,
        nb_training_steps: int,
        last_epoch: int = -1,
    ):
    return LambdaLR(optimizer, lambda i: (
        max(0.0, float(nb_training_steps - i) / nb_training_steps)
    ), last_epoch=last_epoch)

def get_linear_schedule_with_warmup(
        optimizer: Optimizer, 
        nb_warmup_steps: int, 
        nb_training_steps: int, 
        last_epoch: int = -1,
    ):
    return LambdaLR(optimizer, lambda i: (
        float(i) / float(max(1, nb_warmup_steps)) if i < nb_warmup_steps 
        else max(0.0, float(nb_training_steps - i) / float(max(1, nb_training_steps - nb_warmup_steps)))
    ), last_epoch=last_epoch)

def get_cosine_schedule(
        optimizer: Optimizer, 
        nb_training_steps: int, 
        nb_cycles: int = 0.5, 
        last_epoch: int = -1,
    ):
    return LambdaLR(optimizer, lambda i: (
        max(0.0, 0.5 * (1.0 + math.cos(math.pi * nb_cycles * 2.0 * (i / nb_training_steps))))
    ), last_epoch=last_epoch)

def get_cosine_schedule_with_warmup(
        optimizer: Optimizer, 
        nb_warmup_steps: int, 
        nb_training_steps: int, 
        nb_cycles: int = 0.5, 
        last_epoch: int = -1,
    ):
    def lr_lambda(i):
        if i < nb_warmup_steps:
            return i / max(1, nb_warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * nb_cycles * 2.0 * (i / nb_training_steps))))

    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)

def get_cosine_with_hard_restarts_schedule_with_warmup(
        optimizer: Optimizer, 
        nb_warmup_steps: int, 
        nb_training_steps: int, 
        nb_cycles: int = 1, 
        last_epoch: int = -1,
    ):
    def lr_lambda(i):
        if i < nb_warmup_steps:
            return i / max(1, nb_warmup_steps)

        progress = (i - nb_warmup_steps) / max(1, nb_training_steps - nb_warmup_steps)
        if progress >= 1.0:
            return 0.0

        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * ((nb_cycles * progress) % 1.0))))

    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)

def get_polynomial_decay_schedule(
        optimizer: Optimizer,
        nb_training_steps: int,
        lr_end: float = 1e-7,
        power: float = 1.0,
        last_epoch: int = -1,
    ):

    lr_init = optimizer.defaults['lr']
    lr_range = lr_init - lr_end
    def lr_lambda(i):
        if i > nb_training_steps:
            return lr_end / lr_init
        else:
            pct_remaining = 1 - i / nb_training_steps
            decay = lr_range * pct_remaining ** power + lr_end
            return decay / lr_init

    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)

def get_polynomial_decay_schedule_with_warmup(
        optimizer: Optimizer,
        nb_warmup_steps: int,
        nb_training_steps: int,
        lr_end: float = 1e-7,
        power: float = 1.0,
        last_epoch: int = -1,
    ):
    lr_init = optimizer.defaults['lr']
    lr_range = lr_init - lr_end
    def lr_lambda(i):
        if i < nb_warmup_steps:
            return i / max(1, nb_warmup_steps)
        elif i > nb_training_steps:
            return lr_end / lr_init
        else:
            pct_remaining = 1 - (i - nb_warmup_steps) / (nb_training_steps - nb_warmup_steps)
            decay = lr_range * pct_remaining ** power + lr_end
            return decay / lr_init

    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)

TYPE_TO_FUNCTION = {
    SchedulerType.Constant: get_constant_schedule,
    SchedulerType.ConstantWithWarmup: get_constant_schedule_with_warmup,
    SchedulerType.Linear: get_linear_schedule,
    SchedulerType.LinearWithWarmup: get_linear_schedule_with_warmup,
    SchedulerType.Cosine: get_cosine_schedule,
    SchedulerType.CosineWithWarmup: get_cosine_schedule_with_warmup,
    SchedulerType.CosineWithRestartsAndWarmup: get_cosine_with_hard_restarts_schedule_with_warmup,
    SchedulerType.Polynomial: get_polynomial_decay_schedule,
    SchedulerType.PolynomialWithWarmup: get_polynomial_decay_schedule_with_warmup,
}

def get_scheduler_function(scheduler_type: SchedulerType):
    return TYPE_TO_FUNCTION[scheduler_type]

def get_scheduler(
        name: Union[str, SchedulerType],
        optimizer: Optimizer,
        last_epoch: int = -1,
        **scheduler_kwargs,
        # nb_warmup_steps: Optional[int] = None,
        # nb_training_steps: Optional[int] = None,
    ):
    name = SchedulerType(name)
    schedule_fn = get_scheduler_function(name)
    return schedule_fn(optimizer, **scheduler_kwargs, last_epoch = last_epoch)

if __name__ == '__main__':
    net = torch.nn.Linear(128, 1)
    optim = torch.optim.AdamW(net.parameters())
    print(get_scheduler('cosine_with_warmup', optim, nb_warmup_steps=1000, nb_training_steps=200_000, nb_cycles=0.5))
