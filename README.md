# NN Engine Core

[![PyPI version](https://img.shields.io/pypi/v/nn-engine-core?logo=pypi&logoColor=white)](https://pypi.org/project/nn-engine-core/)
[![Python](https://img.shields.io/pypi/pyversions/nn-engine-core?logo=python&logoColor=white)](https://pypi.org/project/nn-engine-core/)
[![Build system](https://img.shields.io/badge/build-scikit--build--core-blue?logo=cmake&logoColor=white)](https://scikit-build-core.readthedocs.io/)
[![Bindings](https://img.shields.io/badge/bindings-pybind11-4C72B0?logo=python&logoColor=white)](https://pybind11.readthedocs.io/)

A high-performance, fully native C++ Neural Network engine exposed to Python via pybind11. 

Designed for rapid experimentation without the Python Global Interpreter Lock (GIL) overhead, `nn-engine-core` executes the entire deep learning training loop (forward pass, validation, loss calculation, backpropagation, and weight updates) strictly in native C++ using Eigen. It utilizes a zero-allocation flat-memory Autograd graph, AVX SIMD vectorization, and dynamically compiled OpenBLAS to achieve massive speedups over mainstream Python frameworks.

## Highlights

- **Native Loop Hoisting**: The `JITCompiler::fit` loop executes entirely in C++, eliminating the Python GIL overhead across epochs and batches.
- **CNN & Modern Layer Support**: Built-in support for `Conv2dLayer` (via parallelized `im2col`), `BatchNorm1dLayer`, `DropoutLayer`, and Leaky ReLUs.
- **Zero-Allocation Autograd**: Uses arena allocation (`Tape`) and flat contiguous memory structs to dynamically build computational graphs without heap allocations.
- **Native Schedulers**: Includes `StepLR` learning rate scheduling calculated natively per-epoch.
- **Pure Python Extensibility**: Easily define custom Autograd operations (`nn.Op`) in pure Python. The engine uses PyBind11 trampolines to dispatch the C++ backward pass dynamically to your Python methods.
- **Native Checkpointing**: Dump and restore raw contiguous memory weights directly to disk via C++ streams (`.nne` files) for blazing fast model saving.

## Installation

Install the released wheel from PyPI (macOS, Linux, and Windows supported):

```bash
pip install nn-engine-core
```

Or install in editable/development mode from the repository (requires CMake 3.18+ and a C++17 compiler):

```bash
pip install -e .
```

## Quick Start: Building a CNN

```python
import numpy as np
import nnengine as nn

# 1. Prepare Data (float32 required)
# 100 samples of 1-channel 8x8 images
X_train = np.random.rand(100, 1 * 8 * 8).astype(np.float32)
y_train = np.eye(10)[np.random.choice(10, 100)].astype(np.float32) # One-hot labels

# 2. Define Network using PyTorch-like Syntax
class MyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Conv2D: 1 in_channel, 16 out_channels, 8x8 input, 3x3 kernel, pad=1
        self.conv1 = self.add_module("conv1", nn.Conv2dLayer(1, 16, 8, 8, kernel_size=3, pad=1))
        self.relu = self.add_module("relu", nn.ReLULayer())
        self.fc = self.add_module("fc", nn.DenseLayer(16 * 8 * 8, 10))

    def forward(self, tape, x):
        x = self.conv1(tape, x)
        x = self.relu(tape, x)
        return self.fc(tape, x)

model = MyCNN()

# 3. Compile & Train using C++ JIT
optimizer = nn.Adam(learning_rate=0.005)
loss_fn = nn.SoftmaxCrossEntropyLoss()
trainer = nn.JITCompiler(model, optimizer, loss_fn)

# Attach a native Learning Rate Scheduler
scheduler = nn.StepLR(optimizer, step_size=20, gamma=0.5)
trainer._cpp_engine.set_scheduler(scheduler)

dataloader = nn.DataLoader(X_train, y_train, batch_size=32)

# Executes entirely in C++ without the GIL!
trainer.fit(dataloader, epochs=40, tol=1e-4)

# 4. Save and Load C++ Binary Checkpoints
model.save_weights("cnn_model.nne")
```

## Defining Custom Autograd Operations in Python

You can easily extend the C++ engine by defining Custom Operations in Pure Python. The C++ `Tape` will safely map NumPy views to the Eigen memory and correctly reverse the graph!

```python
import numpy as np
import nnengine as nn

class MulOp(nn.Op):
    def __init__(self, tape, a, b):
        super().__init__()
        self.a, self.b = a, b
        # Let the C++ Arena allocate the flat memory
        self.out = tape.alloc_tensor(a.data.shape[0], b.data.shape[1], True)

    def forward(self):
        self.out.data = self.a.data * self.b.data 

    def backward(self):
        # Read/Write directly into the C++ Backend via NumPy views!
        if self.a.requires_grad:
            self.a.grad += self.out.grad * self.b.data
        if self.b.requires_grad:
            self.b.grad += self.out.grad * self.a.data
```

## Benchmark Results: NNEngine vs. PyTorch

Because `NNEngine` handles the entire training graph, dataloading, and optimization steps in an isolated C++ environment, it bypasses the heavy Python dispatcher overhead that plagues traditional frameworks on CPU workloads. 

Below is a direct CPU-to-CPU hardware comparison between **NNEngine (C++ JIT)** and **PyTorch (ATen)** utilizing an identical architecture, Adam optimizer, `StepLR` scheduler, and identical mini-batch iterations.

| Dataset | Network Type | NNEngine Acc. | PyTorch Acc. | CPU Speedup |
|---|---|--:|--:|--:|
| **Iris Flower** | MLP | **100.00%** | 100.00% | **~2199x Faster** |
| **Digits** | Conv2D CNN | **98.06%** | 98.61% | **~112x Faster** |
| **Olivetti Faces** | Conv2D CNN | **91.25%** | 87.50% | **~4.5x Faster** |

*Note: For smaller datasets like Iris, the PyTorch Python loop and ATen dispatch latency completely dominate training time. NNEngine executes these loop cycles instantly via raw memory pointers.*