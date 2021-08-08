import sys

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

from model import Net
from ptpt.trainer import TrainerConfig, Trainer

def main():
    # define your train and test datasets
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST('../data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('../data', train=False, download=True, transform=transform)

    # define your model (any `nn.Module`)
    net = Net()

    # define your loss function (calling the model using `self.net`)
    def loss_fn(self, batch):
        X, y = batch
        logits = self.net(X)
        loss = F.nll_loss(logits, y)

        pred = logits.argmax(dim=-1, keepdim=True)
        accuracy = 100. * pred.eq(y.view_as(pred)).sum().item() / y.shape[0]
        return loss, accuracy

    # define the training parameters
    cfg = TrainerConfig(
        exp_name = 'mnist-conv',
        batch_size = 64,
        learning_rate = 4e-4,
        nb_workers = 4,
        save_outputs = False,
        metric_names = ['accuracy']
    )

    # initialise the trainer class
    trainer = Trainer(
        net=net,
        loss_fn=loss_fn,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        cfg=cfg
    )

    # call `trainer.train` to start the training loop
    trainer.train()

    # ..and that's that!

if __name__ == '__main__':
    main()
