# Alex's PyTorch Personal Trainer (ptpt)
> (name subject to change)

This repository contains my personal lightweight framework for deep learning
projects in PyTorch.

## Installation

## Usage

## Examples

An example workflow would go like this:

> Define your training and test datasets:

```python
transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train_dataset = datasets.MNIST('../data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('../data', train=False, download=True, transform=transform)
```

> Define your model:

```python
# in this case, we have imported `Net` from another file
net = Net()
```

> Define your loss function that calls `self.net`, taking the full batch as input:

```python
# minimising classification error
def loss_fn(self, batch):
    X, y = batch
    logits = self.net(X)
    loss = F.nll_loss(logits, y)

    pred = logits.argmax(dim=-1, keepdim=True)
    accuracy = 100. * pred.eq(y.view_as(pred)).sum().item() / y.shape[0]
    return loss, accuracy
```

> Optionally create a configuration object:

```python
# see class definition for full list of parameters
cfg = TrainerConfig(
    exp_name = 'mnist-conv',
    batch_size = 64,
    learning_rate = 4e-4,
    nb_workers = 4,
    save_outputs = False,
    metric_names = ['accuracy']
)
```

> Initialise the Trainer class:

```python
trainer = Trainer(
    net=net,
    loss_fn=loss_fn,
    train_dataset=train_dataset,
    test_dataset=test_dataset,
    cfg=cfg
)
```

> Call `trainer.train()` to begin the training loop

```python
trainer.train() # Go!
```

See more examples [here](examples/).

## Motivation

### Citations

### References
