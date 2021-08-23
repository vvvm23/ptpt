import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

from model import Net
from ptpt.trainer import TrainerConfig, Trainer
from ptpt.callbacks import CallbackType
from ptpt.log import info

def main():
    # define your train and test datasets
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    args = {'root': '../data', 'download': True, 'transform': transform}
    train_dataset = datasets.MNIST(train=True, **args)
    test_dataset = datasets.MNIST(train=False, **args)

    # define your loss function (calling the model using `net`)
    def loss_fn(net, batch):
        X, y = batch
        logits = net(X)
        loss = F.nll_loss(logits, y)

        pred = logits.argmax(dim=-1, keepdim=True)
        accuracy = 100. * pred.eq(y.view_as(pred)).sum().item() / y.shape[0]
        return loss, accuracy

    # define your model (any `nn.Module`)
    net = Net()

    # define the training parameters
    cfg = TrainerConfig(
        exp_dir = '../exp',
        exp_name = 'mnist-conv',
        batch_size = 128,
        learning_rate = 4e-4,
        nb_workers = 4,
        save_outputs = True,
        metric_names = ['accuracy'],
    )

    # initialise the trainer class
    trainer = Trainer(
        net=net,
        loss_fn=loss_fn,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        cfg=cfg,
    )

    # register some callbacks to other behaviour
    def callback_fn(_):
        info("Congratulations, you have completed an epoch!")
    trainer.register_callback(CallbackType.TrainEpoch, callback_fn)

    # call `trainer.train` to start the training loop
    trainer.train()

    # ..and that's that!

if __name__ == '__main__':
    main()
