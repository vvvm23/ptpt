# Alex's PyTorch Framework
> (name subject to change)

This repository contains my personal lightweight framework for deep learning
projects in PyTorch.

## Usage

## Examples

## Motivation

A lot of my deep learning projects tend to repeat a lot of the same patterns.
Some of these patterns are not-so-interesting to me - I'd rather focus on the
more novel aspects of whatever project I am working on. Unfortunately, these
not-so-interesting parts are also of critical importance to the success of any
such project. 

A lightweight framework can help alleviate the strain of repeatedly
reimplementing (or just copy pasting) these patterns. Additionally, it can help
prevent silly mistakes from forgetting very important lines of code - such as
a missing `torch.zero_grad()`.

Additionally, there are lots of nice engineering tricks that are a little hard
to implement. Often, they get ignored completely as they fall outside the
standard deep learning project structure. The framework also aims to abstract
away some commonly used tricks, so that I can benefit from them without even
remembering to include them.

Finally, reproducibility is important in deep learning experiments. I am very
guilty of forgetting to write code to log outputs and parameters. Cue a very
good training run, followed by despair at not being able to reproduce what
I had just done. I'm not a huge fan of logging frameworks like Tensorboard,
preferring either plain-text or a simple binary format,

> Also, fancy logging looks pretty!

I usually work on a machine with only a single GPU, but increasingly I am
finding myself having some kind of access to multiple GPUs. It is currently
a little difficult to adapt code to use distributed training - the recommended
way over single-process, multiple-GPU training. In the future, it would be good
to have support for this.

### Citations

### References
