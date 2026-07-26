from . import _backend

class JITCompiler:
    """High-level Python wrapper around the native JIT training graph.

    The heavy training loop, validation, and early-stopping logic execute in
    C++ through the bound JITGraph object, allowing the GIL to be released 
    while the compiled engine is replaying batches.
    """

    def __init__(self, model, optimizer, loss_fn, regularizer=None):
        """Initialize a compiled training wrapper.

        Args:
            model (nnengine.Module): Layer or Module instance to optimize.
            optimizer (nnengine.Optimizer): Native optimizer implementation.
            loss_fn (nnengine.Loss): Native loss implementation.
            regularizer (nnengine.Regularizer, optional): Optional penalty term.
        """
        self._cpp_engine = _backend.JITGraph(model, optimizer, loss_fn, regularizer)
        self.is_compiled = False
        
    def fit(self, dataloader, epochs, val_dataloader=None, tol=1e-4, n_iter_no_change=10, verbose=True):
        """Train the compiled graph with optional validation and early stopping.

        Args:
            dataloader (nnengine.DataLoader): Native training batch source.
            epochs (int): Number of epochs to train.
            val_dataloader (nnengine.DataLoader, optional): Validation batch source.
            tol (float): Minimum improvement required to reset early stopping.
            n_iter_no_change (int): Number of epochs without improvement before stop.
            verbose (bool): Whether to print progress from the native engine.
        """
        if not self.is_compiled:
            dataloader.reset()
            self._cpp_engine.trace_batch(dataloader)
            self.is_compiled = True
            
        dataloader.reset()
        self._cpp_engine.fast_fit(dataloader, val_dataloader, epochs, tol, n_iter_no_change, verbose)

    def save_checkpoint(self, base_filepath):
        """Save the model weights and optimizer state natively to disk."""
        self._cpp_engine.save_checkpoint(base_filepath)
        
    def load_checkpoint(self, base_filepath):
        """Load the model weights and optimizer state natively from disk."""
        self._cpp_engine.load_checkpoint(base_filepath)