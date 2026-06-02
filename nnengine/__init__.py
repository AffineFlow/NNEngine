"""Python package exports for the NNEngine runtime.

The package exposes the native C++ bindings together with intuitive Python wrappers
for module composition and JIT-compiled training.
"""

from nn_core import (
    Adam, SGD, DenseLayer, ReLULayer, LeakyReLULayer, DropoutLayer, BatchNorm1dLayer,
    MSELoss, SoftmaxCrossEntropyLoss, L2Regularizer, Conv2dLayer, StepLR,
    DataLoader, Tape, Tensor, Op, set_seed
)

from .module import Module
from .compiler import JITCompiler

__all__ = [
    "Module", "JITCompiler", "Adam", "SGD", "DenseLayer", 
    "ReLULayer", "LeakyReLULayer", "DropoutLayer", "BatchNorm1dLayer",
    "MSELoss", "SoftmaxCrossEntropyLoss", "Conv2dLayer", "StepLR",
    "L2Regularizer", "DataLoader", "Tape", "Tensor", "Op", "set_seed"
]