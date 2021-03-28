# FLOP

Pytorch Library for L0 based pruning, as proposed in the paper:
[Structured Pruning of Large Language Models](https://arxiv.org/abs/1910.04732)(EMNLP 2020)

## Install

`pip install -U flop`

## Usage

Create a hard concrete mask of size N:

```python
from flop import HardConrete

N = 100
hardconcrete = HardConcrete(n_in=N)
```

You can then sample masks on the fly with:

```python
mask = hardconcrete()
```

Note that during evaluation, a mask is compiled and fixed.

You may also find these other objects useful:

- ``ProjectedLinear``: replaces a linear layer to include an intermediate projection.
- ``HardConreteProjectedLinear``: the hard conrete version of the ``ProjectedLinear`` module.

You may instantiate the HardConcrete objects directly, or you can choose to first train with
a ``ProjectedLinear`` module, and introduce the hardconcrete mask with:

```python
module = ProjectedLinear(...)
# Perform training

# ...

# Start pruning
pruning_module = HardConcreteProjectedLinear.from_module(module)
```

We also provide some utily functions to replace all ProjectedLinear modules in a model:

```python
from flop import make_hard_concrete

model = make_hard_concrete(model)
```

## Replicate results from the paper

To replicate the SRU numbers, please look at the script ``examples/train_enwik8.py``.

## Cite

```sh
@inproceedings{wang-etal-2020-structured,
    title = "Structured Pruning of Large Language Models",
    author = "Wang, Ziheng  and
      Wohlwend, Jeremy  and
      Lei, Tao",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.emnlp-main.496",
    doi = "10.18653/v1/2020.emnlp-main.496",
    pages = "6151--6162"
}
```
