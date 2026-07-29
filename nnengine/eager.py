from __future__ import annotations

import functools
from typing import Any, Callable, TypeVar

from . import _backend

F = TypeVar("F", bound=Callable[..., Any])

_THREAD_LOCAL_TAPE: _backend.Tape | None = None


def get_eager_tape() -> _backend.Tape:
    global _THREAD_LOCAL_TAPE
    if _THREAD_LOCAL_TAPE is None:
        _THREAD_LOCAL_TAPE = _backend.Tape(record_ops=True)
    return _THREAD_LOCAL_TAPE


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
        ) -> _backend.Tensor:
            tape = get_eager_tape()
            tape.reset()
            optimizer.zero_grad()

            with tape:
                x_tensor = tape.push_tensor(x_batch, False)
                y_tensor = tape.push_tensor(y_batch, False)

                predictions = fn(model, x_tensor, *args, **kwargs)
                loss_val = loss_fn.forward(predictions, y_tensor)
                loss_fn.backward()
                tape.backward()

                if regularizer is not None:
                    loss_val += regularizer.apply(model.parameters())

            optimizer.step()
            return loss_val

        return wrapper  # type: ignore[return-value]

    return decorator