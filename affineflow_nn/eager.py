from __future__ import annotations

import functools
from typing import Any, Callable, TypeVar

from . import _backend

F = TypeVar("F", bound=Callable[..., Any])

def eager_step(
    optimizer: _backend.Optimizer,
    loss_fn: _backend.Loss,
    regularizer: _backend.Regularizer | None = None,
) -> Callable[[F], F]:
    def decorator(fn: F) -> F:
        @functools.wraps(fn)
        def wrapper(
            model: _backend.Module,
            x_batch: Any,
            y_batch: Any,
            *args: Any,
            **kwargs: Any,
        ) -> float:
            optimizer.zero_grad()

            x_tensor = _backend.Tensor(x_batch)
            x_tensor.requires_grad = False
            y_tensor = _backend.Tensor(y_batch)
            y_tensor.requires_grad = False

            predictions = fn(model, x_tensor, *args, **kwargs)
            loss_val = loss_fn.forward(predictions, y_tensor)
            
            # Triggers C++ backpropagation and tape reset
            loss_fn.backward()

            if regularizer is not None:
                loss_val += regularizer.apply(model.parameters())

            optimizer.step()
            return loss_val

        return wrapper

    return decorator