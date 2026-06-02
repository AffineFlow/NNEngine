"""Python package exports for the NNEngine runtime.

The package exposes the native C++ bindings together with small Python wrappers
for module composition and JIT training.
"""

import nn_core

from nn_core import (
    Adam, SGD, DenseLayer, ReLULayer, LeakyReLULayer, DropoutLayer, BatchNorm1dLayer,
    MSELoss, SoftmaxCrossEntropyLoss, L2Regularizer,
    DataLoader, Tape, Tensor, Op, set_seed
)

from .module import Module
from .compiler import JITCompiler

__all__ = [
    "Module", "JITCompiler", "Adam", "SGD", "DenseLayer", 
    "ReLULayer", "LeakyReLULayer", "DropoutLayer", "BatchNorm1dLayer",
    "MSELoss", "SoftmaxCrossEntropyLoss", 
    "L2Regularizer", "DataLoader", "Tape", "Tensor", "Op", "set_seed"
]