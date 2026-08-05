from __future__ import annotations

from . import _backend


class JITCompiler:
    """High-level Python wrapper around the native JIT training graph."""

    def __init__(
        self,
        model: _backend.Module,
        optimizer: _backend.Optimizer,
        loss_fn: _backend.Loss,
        regularizer: _backend.Regularizer | None = None,
    ) -> None:
        self._cpp_engine = _backend.JITGraph(model, optimizer, loss_fn, regularizer)
        self.is_compiled: bool = False

    def fit(
        self,
        dataloader: _backend.DataLoader,
        epochs: int,
        val_dataloader: _backend.DataLoader | None = None,
        tol: float = 1e-4,
        n_iter_no_change: int = 10,
        verbose: bool = True,
    ) -> None:
        if not self.is_compiled:
            dataloader.reset()
            self._cpp_engine.trace_batch(dataloader)
            self.is_compiled = True

        dataloader.reset()
        self._cpp_engine.fast_fit(
            dataloader, val_dataloader, epochs, tol, n_iter_no_change, verbose
        )

    def save_checkpoint(self, base_filepath: str) -> None:
        self._cpp_engine.save_checkpoint(base_filepath)

    def load_checkpoint(self, base_filepath: str) -> None:
        self._cpp_engine.load_checkpoint(base_filepath)