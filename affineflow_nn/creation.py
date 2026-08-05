import numpy as np
from . import _backend

def zeros(*shape, requires_grad=False) -> _backend.Tensor:
    """Returns a tensor filled with the scalar value 0."""
    t = _backend.Tensor(np.zeros(shape, dtype=np.float32))
    t.requires_grad = requires_grad
    return t

def ones(*shape, requires_grad=False) -> _backend.Tensor:
    """Returns a tensor filled with the scalar value 1."""
    t = _backend.Tensor(np.ones(shape, dtype=np.float32))
    t.requires_grad = requires_grad
    return t

def randn(*shape, requires_grad=False) -> _backend.Tensor:
    """Returns a tensor filled with random numbers from a standard normal distribution."""
    t = _backend.Tensor(np.random.randn(*shape).astype(np.float32))
    t.requires_grad = requires_grad
    return t