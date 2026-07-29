from typing import Any

from . import _backend


class Module(_backend.Module):
    """Python base class for composing NNEngine layers."""

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