# Alex's PyTorch Personal Trainer (pt pt)
> (name subject to change)

This repository contains my personal lightweight framework for deep learning
projects in Torch.

## Installation

## Usage
The core of the library is the `trainer.Trainer` class.

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
I found myself repeating a lot of same structure in many of my deep learning
projects. This project is the culmination of my efforts refining the typical
structure of my projects into (what I hope to be) a wholly reusable and 
general-purpose library.

Additionally, there are many nice theoretical and engineering tricks that
are available to deep learning researchers. Unfortunately, a lot of them are 
forgotten because they fall outside the typical workflow, despite them being
very beneficial to include. Another goal of this project is to transparently
include these tricks so they can be added and removed with minimal code change.
Where it is sane to do so, some of these could be on by default.

Finally, I am guilty of forgetting to implement decent logging: both of 
standard output and of metrics. Logging of standard output is not hard, and 
is implemented using other libraries such as [rich](https://github.com/willmcgugan/rich).
However, metric logging is less obvious. I'd like to avoid larger dependencies 
such as tensorboard being an integral part of the project, so metrics will be
logged to simple numpy arrays. The library will then provide functions to 
produce plots from these, or they can be used in another library.

### TODO:

- [ ] A lot.

### Citations

### References
