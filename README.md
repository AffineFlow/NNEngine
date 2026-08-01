# NNEngine

[![PyPI version](https://img.shields.io/pypi/v/nn-engine-core?logo=pypi&logoColor=white)](https://pypi.org/project/nn-engine-core/)
[![Python](https://img.shields.io/pypi/pyversions/nn-engine-core?logo=python&logoColor=white)](https://pypi.org/project/nn-engine-core/)
[![Build system](https://img.shields.io/badge/build-scikit--build--core-blue?logo=cmake&logoColor=white)](https://scikit-build-core.readthedocs.io/)
[![Bindings](https://img.shields.io/badge/bindings-pybind11-4C72B0?logo=python&logoColor=white)](https://pybind11.readthedocs.io/)

A high-performance, fully native C++ Neural Network engine exposed to Python via pybind11. 

Designed for rapid experimentation without the Python Global Interpreter Lock (GIL) overhead, `nn-engine-core` executes the entire deep learning training loop strictly in native C++ using Eigen. It utilizes a zero-allocation flat-memory Autograd graph, AVX SIMD vectorization, and dynamically compiled OpenBLAS to achieve massive speedups over mainstream Python frameworks.

## Highlights

- **Dual Execution Modes**: Train dynamic graphs using the `@nne.eager` decorator for zero-allocation step-by-step execution, or compile static graphs with `JITCompiler` for maximum throughput.
- **Dynamic Hardware Dispatch**: Automatically detects your CPU at runtime to dispatch highly optimized AVX2 or AVX-512 instructions.
- **CNN & Modern Layer Support**: Built-in support for `Conv2dLayer`, `MaxPool2dLayer`, `AvgPool2dLayer`, `FlattenLayer`, `BatchNorm1dLayer`, `DropoutLayer`, and Leaky ReLUs.
- **Zero-Allocation Autograd**: Uses arena allocation (`Tape`) and flat contiguous memory structs to dynamically build computational graphs without heap allocations.

## Installation

Install the released wheel from PyPI:

    pip install nn-engine-core

## Quick Start: Building a CNN

```py
import numpy as np
import nnengine as nne

X_train = np.random.rand(100, 1, 64, 64).astype(np.float32)
y_train = np.eye(40, dtype=np.float32)[np.random.choice(40, 100)]

class NNEngineDeepCNN(nne.Module):
    def __init__(self, in_h, in_w, num_classes):
        super().__init__()
        self.conv1 = nne.Conv2dLayer(1, 16, in_h, in_w, kernel_size=5, stride=1, pad=2)
        self.act1 = nne.LeakyReLULayer(0.01)
        self.pool1 = nne.MaxPool2dLayer(16, in_h, in_w, kernel_size=2, stride=2, pad=0)
        
        out_h1, out_w1 = in_h // 2, in_w // 2
        self.conv2 = nne.Conv2dLayer(16, 32, out_h1, out_w1, kernel_size=3, stride=1, pad=1)
        self.act2 = nne.LeakyReLULayer(0.01)
        self.pool2 = nne.MaxPool2dLayer(32, out_h1, out_w1, kernel_size=2, stride=2, pad=0)
        
        out_h2, out_w2 = out_h1 // 2, out_w1 // 2
        self.flatten = nne.FlattenLayer()
        self.fc = nne.DenseLayer(32 * out_h2 * out_w2, num_classes)

    def forward(self, x):
        x = self.pool1(self.act1(self.conv1(x)))
        x = self.pool2(self.act2(self.conv2(x)))
        x = self.flatten(x)
        return self.fc(x)

model = NNEngineDeepCNN(64, 64, 40)
optimizer = nne.Adam(learning_rate=0.001)
optimizer.set_parameters(model.parameters())
loss_fn = nne.SoftmaxCrossEntropyLoss()

trainer = nne.JITCompiler(model, optimizer, loss_fn)
dataloader = nne.DataLoader(X_train, y_train, batch_size=32, shuffle=True, drop_last=True)
trainer.fit(dataloader, epochs=40, verbose=False)
trainer.save_checkpoint("faces_model")
```

## Latest Benchmark Results

| Benchmark Task | Scikit-Learn | PyTorch (ATen) | NNEngine (Eager) | NNEngine (C++ JIT) | Max Speedup (vs PyTorch) |
|---|--:|--:|--:|--:|--:|
| **Deep Regression (Housing)** | MSE: 0.2811<br>Time: 4.94s | MSE: 0.2720<br>Time: 13.32s | MSE: 0.2662<br>**Time: 2.47s** | MSE: 0.2669<br>Time: 4.02s | **5.39x** |
| **Deep CNN (Faces)** | *N/A* | Acc: 98.75%<br>Time: 15.19s | Acc: 98.75%<br>Time: 12.17s | Acc: 98.75%<br>**Time: 11.22s** | **1.35x** |