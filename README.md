# NNEngine

[![PyPI version](https://img.shields.io/pypi/v/nn-engine-core?logo=pypi&logoColor=white)](https://pypi.org/project/nn-engine-core/)
[![Python](https://img.shields.io/pypi/pyversions/nn-engine-core?logo=python&logoColor=white)](https://pypi.org/project/nn-engine-core/)
[![Build system](https://img.shields.io/badge/build-scikit--build--core-blue?logo=cmake&logoColor=white)](https://scikit-build-core.readthedocs.io/)
[![Bindings](https://img.shields.io/badge/bindings-pybind11-4C72B0?logo=python&logoColor=white)](https://pybind11.readthedocs.io/)

A high-performance, fully native C++ Neural Network engine exposed to Python via pybind11. 

Designed for rapid experimentation without the Python Global Interpreter Lock (GIL) overhead, `nn-engine-core` executes the entire deep learning training loop (forward pass, validation, loss calculation, backpropagation, and weight updates) strictly in native C++ using Eigen. It utilizes a zero-allocation flat-memory Autograd graph, AVX SIMD vectorization, and dynamically compiled OpenBLAS to achieve massive speedups over mainstream Python frameworks.

## Highlights

- **Dual Execution Modes**: Train dynamic graphs using the `@nne.eager` decorator for zero-allocation step-by-step execution, or compile static graphs with `JITCompiler` for maximum throughput on heavy workloads.
- **Dynamic Hardware Dispatch**: The engine ships as a "fat wheel," automatically detecting your CPU at runtime to dispatch highly optimized AVX2 or AVX-512 instructions.
- **Native Loop Hoisting**: The `JITCompiler::fit` loop executes entirely in C++, eliminating the Python GIL overhead across epochs and batches.
- **CNN & Modern Layer Support**: Built-in support for `Conv2dLayer` (via parallelized `im2col`), `BatchNorm1dLayer`, `DropoutLayer`, and Leaky ReLUs.
- **Zero-Allocation Autograd**: Uses arena allocation (`Tape`) and flat contiguous memory structs to dynamically build computational graphs without heap allocations.

## Installation

Install the released wheel from PyPI (macOS, Linux, and Windows supported):

    pip install nn-engine-core

Or install in editable/development mode from the repository (requires CMake 3.18+ and a C++17 compiler):

    pip install -e .

## Quick Start: Building a CNN

    import numpy as np
    import nnengine as nne

    X_train = np.random.rand(100, 1, 64, 64).astype(np.float32)
    y_train = np.eye(40, dtype=np.float32)[np.random.choice(40, 100)]

    class NNEngineDeepCNN(nne.Module):
        def __init__(self, in_h, in_w, num_classes):
            super().__init__()
            self.conv1 = nne.Conv2dLayer(1, 16, in_h, in_w, kernel_size=5, stride=2, pad=2)
            self.act1 = nne.LeakyReLULayer(0.01)
            out_h1, out_w1 = in_h // 2, in_w // 2
            self.conv2 = nne.Conv2dLayer(16, 32, out_h1, out_w1, kernel_size=3, stride=2, pad=1)
            self.act2 = nne.LeakyReLULayer(0.01)
            out_h2, out_w2 = out_h1 // 2, out_w1 // 2
            self.fc = nne.DenseLayer(32 * out_h2 * out_w2, num_classes)

        def forward(self, x):
            x = self.act1(self.conv1(x))
            x = self.act2(self.conv2(x))
            return self.fc(x)

    model = NNEngineDeepCNN(64, 64, 40)
    optimizer = nne.Adam(learning_rate=0.001)
    loss_fn = nne.SoftmaxCrossEntropyLoss()
    trainer = nne.JITCompiler(model, optimizer, loss_fn)

    dataloader = nne.DataLoader(X_train, y_train, batch_size=32, shuffle=True, drop_last=True)
    trainer.fit(dataloader, epochs=40, verbose=False)
    trainer.save_checkpoint("faces_model")

## Quick Start: Eager Mode Training

If you prefer PyTorch-style step-by-step execution or need dynamic control flow (like `if` statements inside your forward pass), use the `@nne.eager` decorator. It manages the C++ Autograd tape and GIL releases automatically:

```python
import nnengine as nne

model = nne.DenseLayer(10, 1)
optimizer = nne.AdamW(learning_rate=0.01, weight_decay=1e-4)
optimizer.set_parameters(model.parameters())
loss_fn = nne.MSELoss()

@nne.eager(optimizer=optimizer, loss_fn=loss_fn)
def train_step(mod, x):
    # Standard Python control flow works perfectly here!
    return mod(x)

# Training loop
for epoch in range(10):
    dataloader.reset()
    while dataloader.has_next():
        dataloader.next_batch(X_batch, y_batch)
        loss = train_step(model, X_batch, y_batch)
```

## Defining Custom Autograd Operations in Python

    import numpy as np
    import nnengine as nne

    class MulOp(nne.Op):
        def __init__(self, a, b):
            super().__init__()
            self.a = a
            self.b = b

        def forward(self):
            self.out_data = self.a.data * self.b.data

        def backward(self):
            if self.a.requires_grad:
                self.a.grad += self.out.grad * self.b.data
            if self.b.requires_grad:
                self.b.grad += self.out.grad * self.a.data

## Testing & Validation Suite

Execute the full Scikit-Learn dataset validation and benchmark suite via:

    python examples/script.py

To benchmark raw FLOPs and OpenMP scaling without network I/O overhead, execute the massive synthetic stress test:

    python examples/stress_test.py

### Latest Benchmark Results

NNEngine's native C++ execution eliminates the Python GIL, consistently outperforming PyTorch's ATen backend on the CPU in both JIT and Eager modes.

| Benchmark Task | Scikit-Learn | PyTorch (ATen) | NNEngine (Eager) | NNEngine (C++ JIT) | Max Speedup (vs PyTorch) |
|---|--:|--:|--:|--:|--:|
| **Deep Regression (Housing)** | MSE: 0.2811<br>Time: 8.59s | MSE: 0.2720<br>Time: 18.18s | MSE: 0.2662<br>**Time: 3.19s** | MSE: 0.2669<br>Time: 5.10s | **5.69x** |
| **Deep CNN (Faces)** | *N/A* | Acc: 97.50%<br>Time: 5.97s | Acc: 97.50%<br>Time: 5.02s | Acc: 97.50%<br>**Time: 4.41s** | **1.35x** |
| **Massive Wide MLP** *(Stress)* | *N/A* | Time: 46.09s | Time: 38.98s | **Time: 38.09s** | **1.21x** |
| **Deep Spatial CNN** *(Stress)* | *N/A* | Time: 39.96s | Time: 36.76s | **Time: 35.70s** | **1.12x** |
