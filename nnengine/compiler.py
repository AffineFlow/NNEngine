import nn_core

class JITCompiler:
    """High-level Python wrapper around the native JIT training graph."""

    def __init__(self, model, optimizer, loss_fn, regularizer=None):
        self._cpp_engine = nn_core.JITGraph(model, optimizer, loss_fn, regularizer)
        self.is_compiled = False
        
    def fit(self, dataloader, epochs, val_dataloader=None, tol=1e-4, n_iter_no_change=10, verbose=True):
        if not self.is_compiled:
            self._cpp_engine.trace_batch(dataloader)
            self.is_compiled = True
            self._cpp_engine.fast_loop(dataloader)
            epochs -= 1
        if epochs > 0:
            self._cpp_engine.fast_fit(dataloader, val_dataloader, epochs, tol, n_iter_no_change, verbose)

    def save_checkpoint(self, base_filepath):
        """Save the model weights and optimizer state natively to disk.
        
        This will generate two files: 
        - <base_filepath>.weights.nne
        - <base_filepath>.opt.nne
        """
        self._cpp_engine.save_checkpoint(base_filepath)
        
    def load_checkpoint(self, base_filepath):
        """Load the model weights and optimizer state natively from disk."""
        self._cpp_engine.load_checkpoint(base_filepath)