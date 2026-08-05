from typing import Any

from . import _backend


class Module(_backend.Module):
    """Python base class for composing AffineFlow-NN layers."""

    def __init__(self) -> None:
        super().__init__()

    def __setattr__(self, name: str, value: Any) -> None:
        if isinstance(value, (_backend.Layer, _backend.Module)):
            if hasattr(self, name):
                raise AttributeError(
                    f"Layer '{name}' is already registered. Reassignment is not allowed."
                )
            self.register_module(name, value)
        super().__setattr__(name, value)

    def __repr__(self):
        lines = [f"{self.__class__.__name__}("]
        for key, val in self.__dict__.items():
            if hasattr(val, 'parameters') and callable(val): 
                val_str = repr(val).replace('\n', '\n  ')
                lines.append(f"  ({key}): {val_str}")
        lines.append(")")
        return "\n".join(lines)