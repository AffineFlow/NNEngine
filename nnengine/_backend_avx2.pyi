"""
C++ Core Backend Engine for NNEngine
"""
from __future__ import annotations
import collections.abc
import numpy
import numpy.typing
import typing
__all__: list[str] = ['Adam', 'AdamW', 'BatchNorm1dLayer', 'Conv2dLayer', 'DataLoader', 'DenseLayer', 'DropoutLayer', 'JITGraph', 'L2Regularizer', 'Layer', 'LeakyReLULayer', 'Loss', 'MSELoss', 'Module', 'Op', 'Optimizer', 'ReLULayer', 'Regularizer', 'SGD', 'Scheduler', 'SoftmaxCrossEntropyLoss', 'StepLR', 'Tape', 'Tensor', 'set_seed']
class Adam(Optimizer):
    """
    Adaptive Moment Estimation optimizer.
    """
    def __init__(self, learning_rate: typing.SupportsFloat | typing.SupportsIndex = 0.0010000000474974513) -> None:
        """
        Initialize Adam with learning rate.
        """
class AdamW(Optimizer):
    """
    Adam optimizer with decoupled weight decay.
    """
    def __init__(self, learning_rate: typing.SupportsFloat | typing.SupportsIndex = 0.0010000000474974513, weight_decay: typing.SupportsFloat | typing.SupportsIndex = 0.009999999776482582) -> None:
        """
        Initialize AdamW with learning rate and decay penalty.
        """
class BatchNorm1dLayer(Layer):
    """
    Applies Batch Normalization over a 1D input tensor.
    """
    def __call__(self, input: Tensor) -> Tensor:
        """
        Alias for forward.
        """
    def __init__(self, num_features: typing.SupportsInt | typing.SupportsIndex, eps: typing.SupportsFloat | typing.SupportsIndex = 9.999999747378752e-06, momentum: typing.SupportsFloat | typing.SupportsIndex = 0.10000000149011612) -> None:
        """
        Initialize batch normalization.
        """
    def forward(self, input: Tensor) -> Tensor:
        """
        Apply batch normalization.
        """
class Conv2dLayer(Layer):
    """
    Applies a 2D convolution over an input signal.
    """
    def __call__(self, input: Tensor) -> Tensor:
        """
        Alias for forward.
        """
    def __init__(self, in_channels: typing.SupportsInt | typing.SupportsIndex, out_channels: typing.SupportsInt | typing.SupportsIndex, in_h: typing.SupportsInt | typing.SupportsIndex, in_w: typing.SupportsInt | typing.SupportsIndex, kernel_size: typing.SupportsInt | typing.SupportsIndex, stride: typing.SupportsInt | typing.SupportsIndex = 1, pad: typing.SupportsInt | typing.SupportsIndex = 0) -> None:
        """
        Initialize spatial convolution.
        """
    def forward(self, input: Tensor) -> Tensor:
        """
        Apply 2D convolution.
        """
class DataLoader:
    """
    Batch generator for training and evaluation datasets.
    """
    def __init__(self, X: Tensor, y: Tensor, batch_size: typing.SupportsInt | typing.SupportsIndex, shuffle: bool = True, drop_last: bool = False) -> None:
        """
        Initialize a data loader with features and targets.
        """
    def has_next(self) -> bool:
        """
        Check if there are remaining batches in the current epoch.
        """
    def next_batch(self, X_batch: Tensor, y_batch: Tensor) -> None:
        """
        Populate the provided tensors with the next batch of data.
        """
    def reset(self) -> None:
        """
        Reset the loader to the beginning of the dataset and optionally reshuffle.
        """
class DenseLayer(Layer):
    """
    Fully connected affine layer with learned weights and bias.
    """
    def __call__(self, input: Tensor) -> Tensor:
        """
        Alias for forward.
        """
    def __init__(self, input_dim: typing.SupportsInt | typing.SupportsIndex, output_dim: typing.SupportsInt | typing.SupportsIndex) -> None:
        """
        Initialize with Glorot uniform weights.
        """
    def forward(self, input: Tensor) -> Tensor:
        """
        Apply the affine transform.
        """
class DropoutLayer(Layer):
    """
    Randomly zeroes elements of the input tensor during training.
    """
    def __call__(self, input: Tensor) -> Tensor:
        """
        Alias for forward.
        """
    def __init__(self, p: typing.SupportsFloat | typing.SupportsIndex = 0.5) -> None:
        """
        Probability of an element to be zeroed.
        """
    def forward(self, input: Tensor) -> Tensor:
        """
        Apply dropout mask.
        """
class JITGraph:
    """
    High-performance compiled training loop executor.
    """
    def __init__(self, model: Layer, optimizer: Optimizer, loss_fn: Loss, regularizer: Regularizer = None) -> None:
        """
        Bind the model, optimizer, and loss function to the JIT compiler.
        """
    def evaluate(self, dataloader: DataLoader) -> float:
        """
        Evaluate the model without accumulating gradients.
        """
    def fast_fit(self, dataloader: DataLoader, val_dataloader: DataLoader = None, epochs: typing.SupportsInt | typing.SupportsIndex, tol: typing.SupportsFloat | typing.SupportsIndex = 9.999999747378752e-05, n_iter_no_change: typing.SupportsInt | typing.SupportsIndex = 10, verbose: bool = True) -> None:
        """
        Train the JIT graph natively with early stopping.
        """
    def fast_loop(self, dataloader: DataLoader) -> tuple[float, int]:
        """
        Run a full epoch natively.
        """
    def load_checkpoint(self, base_filepath: str) -> None:
        """
        Load engine state from disk.
        """
    def save_checkpoint(self, base_filepath: str) -> None:
        """
        Save engine state to disk.
        """
    def set_scheduler(self, scheduler: Scheduler) -> None:
        """
        Attach a learning rate scheduler to the compiled graph.
        """
    def trace_batch(self, dataloader: DataLoader) -> float:
        """
        Trace the autograd graph for a single batch.
        """
class L2Regularizer(Regularizer):
    """
    L2 weight decay regularizer.
    """
    def __init__(self, l2: typing.SupportsFloat | typing.SupportsIndex = 9.999999747378752e-05) -> None:
        """
        Construct with a scaling penalty coefficient.
        """
class Layer:
    """
    Abstract building block for differentiable model components.
    """
    def __call__(self, input: Tensor) -> Tensor:
        """
        Alias for forward.
        """
    def eval(self) -> None:
        """
        Set the layer to evaluation mode.
        """
    def forward(self, input: Tensor) -> Tensor:
        """
        Run the layer on tape-owned input storage.
        """
    def parameters(self) -> list[Tensor]:
        """
        Return mutable pointers to trainable parameters.
        """
    def train(self, mode: bool = True) -> None:
        """
        Set the layer to training mode (enables dropout, batchnorm tracking, etc).
        """
class LeakyReLULayer(Layer):
    """
    Elementwise leaky rectified linear activation.
    """
    def __call__(self, input: Tensor) -> Tensor:
        """
        Alias for forward.
        """
    def __init__(self, alpha: typing.SupportsFloat | typing.SupportsIndex = 0.009999999776482582) -> None:
        """
        Construct with a given negative slope.
        """
    def forward(self, input: Tensor) -> Tensor:
        """
        Apply Leaky ReLU.
        """
class Loss:
    """
    Objective function that produces a scalar training signal.
    """
    def backward(self) -> None:
        """
        Explicitly seed the gradient into the prediction tensor.
        """
    def forward(self, predictions: Tensor, targets: Tensor) -> float:
        """
        Bind tensors and compute the forward loss scalar.
        """
class MSELoss(Loss):
    """
    Mean-squared-error objective for regression.
    """
    def __init__(self) -> None:
        """
        Initialize MSELoss.
        """
class Module(Layer):
    """
    Base class for composing layers into a network.
    """
    def __call__(self, input: Tensor) -> Tensor:
        """
        Alias for forward.
        """
    def __init__(self) -> None:
        ...
    def forward(self, input: Tensor) -> Tensor:
        """
        Execute the forward pass of the module.
        """
    def load_weights(self, filepath: str) -> None:
        """
        Load serialized weights from disk.
        """
    def named_parameters(self) -> dict[str, Tensor]:
        """
        Return a dictionary mapping parameter names to Tensors.
        """
    def parameters(self) -> list[Tensor]:
        """
        Return all registered parameters recursively.
        """
    def predict(self, X: Tensor) -> Tensor:
        """
        Run a detached, GIL-released inference pass.
        """
    def register_module(self, name: str, layer: Layer) -> Layer:
        """
        Register a child layer or module natively.
        """
    def save_weights(self, filepath: str) -> None:
        """
        Serialize module weights to disk.
        """
class Op:
    """
    Base interface for a differentiable primitive.
    """
    def __init__(self) -> None:
        ...
    def backward(self) -> None:
        """
        Accumulate gradients for the primitive inputs.
        """
    def forward(self) -> None:
        """
        Execute the forward pass for the primitive.
        """
class Optimizer:
    """
    Base class for parameter update rules.
    """
    def set_parameters(self, params: collections.abc.Sequence[Tensor]) -> None:
        """
        Bind the trainable tensors to the optimizer.
        """
    def step(self) -> None:
        """
        Execute a single optimization step.
        """
    def zero_grad(self) -> None:
        """
        Clear the gradients of all optimized parameters.
        """
class ReLULayer(Layer):
    """
    Elementwise rectified linear activation.
    """
    def __call__(self, input: Tensor) -> Tensor:
        """
        Alias for forward.
        """
    def __init__(self) -> None:
        ...
    def forward(self, input: Tensor) -> Tensor:
        """
        Apply ReLU.
        """
class Regularizer:
    """
    Penalty term applied to trainable parameters during optimization.
    """
    def apply(self, parameters: collections.abc.Sequence[Tensor]) -> float:
        """
        Accumulate a regularization penalty and any gradient adjustment.
        """
class SGD(Optimizer):
    """
    Plain stochastic gradient descent optimizer.
    """
    def __init__(self, learning_rate: typing.SupportsFloat | typing.SupportsIndex = 0.009999999776482582) -> None:
        """
        Initialize SGD with learning rate.
        """
class Scheduler:
    """
    Base class for learning rate scheduling.
    """
class SoftmaxCrossEntropyLoss(Loss):
    """
    Numerically stable softmax cross-entropy for classification.
    """
    def __init__(self) -> None:
        """
        Initialize SoftmaxCrossEntropyLoss.
        """
class StepLR(Scheduler):
    """
    Decays the learning rate by gamma every step_size epochs.
    """
    def __init__(self, optimizer: Optimizer, step_size: typing.SupportsInt | typing.SupportsIndex, gamma: typing.SupportsFloat | typing.SupportsIndex = 0.10000000149011612) -> None:
        """
        Initialize the StepLR scheduler.
        """
    def step(self) -> None:
        """
        Update the optimizer's learning rate.
        """
class Tape:
    """
    Context manager for recording operations for autograd.
    """
    def __enter__(self) -> Tape:
        ...
    def __exit__(self, arg0: typing.Any, arg1: typing.Any, arg2: typing.Any) -> None:
        ...
    def __init__(self, record_ops: bool = True) -> None:
        ...
    def alloc_tensor(self, shape: collections.abc.Sequence[typing.SupportsInt | typing.SupportsIndex], requires_grad: bool = True) -> Tensor:
        """
        Allocate a zeroed tensor from the tape's memory pool.
        """
    def backward(self) -> None:
        """
        Execute the reverse-mode accumulation (backpropagation).
        """
    def push_tensor(self, data: Tensor, requires_grad: bool = True) -> Tensor:
        """
        Push an existing tensor onto the tape, copying its data.
        """
    def reset(self) -> None:
        """
        Clear the tape's recorded ops and reset the memory pool index.
        """
    @property
    def record_ops(self) -> bool:
        """
        Whether the tape is actively recording operations.
        """
    @record_ops.setter
    def record_ops(self, arg1: bool) -> None:
        ...
class Tensor:
    """
    A multi-dimensional array with autograd support.
    """
    @typing.overload
    def __add__(self, other: Tensor) -> typing.Any:
        """
        Element-wise addition of two tensors.
        """
    @typing.overload
    def __add__(self, val: typing.SupportsFloat | typing.SupportsIndex) -> typing.Any:
        ...
    def __buffer__(self, flags):
        """
        Return a buffer object that exposes the underlying memory of the object.
        """
    @typing.overload
    def __iadd__(self, other: Tensor) -> Tensor:
        """
        In-place addition (detached from autograd).
        """
    @typing.overload
    def __iadd__(self, val: typing.SupportsFloat | typing.SupportsIndex) -> Tensor:
        ...
    @typing.overload
    def __imul__(self, other: Tensor) -> Tensor:
        """
        In-place multiplication (detached from autograd).
        """
    @typing.overload
    def __imul__(self, val: typing.SupportsFloat | typing.SupportsIndex) -> Tensor:
        ...
    def __init__(self, data: typing.Annotated[numpy.typing.ArrayLike, numpy.float32]) -> None:
        """
        Initialize a tensor from a NumPy array.
        """
    @typing.overload
    def __isub__(self, other: Tensor) -> Tensor:
        """
        In-place subtraction (detached from autograd).
        """
    @typing.overload
    def __isub__(self, val: typing.SupportsFloat | typing.SupportsIndex) -> Tensor:
        ...
    def __matmul__(self, other: Tensor) -> typing.Any:
        """
        Matrix multiplication of two 2D tensors.
        """
    @typing.overload
    def __mul__(self, other: Tensor) -> typing.Any:
        """
        Element-wise multiplication of two tensors.
        """
    @typing.overload
    def __mul__(self, val: typing.SupportsFloat | typing.SupportsIndex) -> typing.Any:
        ...
    def __radd__(self, val: typing.SupportsFloat | typing.SupportsIndex) -> typing.Any:
        ...
    def __release_buffer__(self, buffer):
        """
        Release the buffer object that exposes the underlying memory of the object.
        """
    def __repr__(self) -> str:
        ...
    def __rmul__(self, val: typing.SupportsFloat | typing.SupportsIndex) -> typing.Any:
        ...
    def __rsub__(self, val: typing.SupportsFloat | typing.SupportsIndex) -> typing.Any:
        ...
    def __rtruediv__(self, val: typing.SupportsFloat | typing.SupportsIndex) -> typing.Any:
        ...
    @typing.overload
    def __sub__(self, other: Tensor) -> typing.Any:
        """
        Element-wise subtraction of two tensors.
        """
    @typing.overload
    def __sub__(self, val: typing.SupportsFloat | typing.SupportsIndex) -> typing.Any:
        ...
    @typing.overload
    def __truediv__(self, other: Tensor) -> typing.Any:
        """
        Element-wise division of two tensors.
        """
    @typing.overload
    def __truediv__(self, val: typing.SupportsFloat | typing.SupportsIndex) -> typing.Any:
        ...
    def backward(self) -> None:
        """
        Computes the gradient of current tensor w.r.t. graph leaves.
        """
    def exp(self) -> typing.Any:
        """
        Element-wise exponential.
        """
    def log(self) -> typing.Any:
        """
        Element-wise natural logarithm.
        """
    def mean(self) -> typing.Any:
        """
        Mean of all elements.
        """
    def relu(self) -> typing.Any:
        """
        Element-wise rectified linear activation.
        """
    def sum(self) -> typing.Any:
        """
        Sum of all elements.
        """
    def transpose(self) -> typing.Any:
        """
        Transposes a 2D tensor.
        """
    @property
    def T(self) -> typing.Any:
        """
        Returns a view of the 2D tensor with its dimensions reversed.
        """
    @property
    def data(self) -> numpy.typing.NDArray[numpy.float32]:
        """
        Zero-copy NumPy view of the underlying forward data.
        """
    @property
    def grad(self) -> numpy.typing.NDArray[numpy.float32]:
        """
        Zero-copy NumPy view of the underlying gradient data.
        """
    @property
    def requires_grad(self) -> bool:
        """
        Whether this tensor should accumulate gradients during the backward pass.
        """
    @requires_grad.setter
    def requires_grad(self, arg1: bool) -> None:
        ...
    @property
    def shape(self) -> list[int]:
        """
        The dimensions of the tensor.
        """
    @shape.setter
    def shape(self, arg1: collections.abc.Sequence[typing.SupportsInt | typing.SupportsIndex]) -> None:
        ...
def set_seed(seed: typing.SupportsInt | typing.SupportsIndex) -> None:
    """
    Seed the shared RNGs to make initialization, shuffling, and dropout repeatable.
    """
