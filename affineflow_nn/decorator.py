from __future__ import annotations

from typing import Callable, TypeVar

from . import _backend
from .compiler import JITCompiler
from .eager import eager_step

F = TypeVar("F", bound=Callable[..., "_backend.Tensor"])


def eager(
    fn: F | None = None,
    *,
    optimizer: _backend.Optimizer | None = None,
    loss_fn: _backend.Loss | None = None,
    regularizer: _backend.Regularizer | None = None,
) -> Callable[..., "_backend.Tensor"]:
    if fn is None:
        return lambda f: eager_step(optimizer, loss_fn, regularizer)(f)
    return eager_step(optimizer, loss_fn, regularizer)(fn)


def jit(
    model: _backend.Module,
    optimizer: _backend.Optimizer,
    loss_fn: _backend.Loss,
    regularizer: _backend.Regularizer | None = None,
) -> JITCompiler:
    return JITCompiler(model, optimizer, loss_fn, regularizer)