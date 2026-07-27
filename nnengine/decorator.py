from .compiler import JITCompiler
from .eager import eager_step

def eager(fn=None, *, optimizer=None, loss_fn=None, regularizer=None):
    """
    Decorator to mark a function for high-performance zero-allocation eager execution.
    Automatically manages the thread-local tape context, batch injection, and optimizer stepping.
    """
    if fn is None:
        return lambda f: eager_step(optimizer, loss_fn, regularizer)(f)
    return eager_step(optimizer, loss_fn, regularizer)(fn)

def jit(model, optimizer, loss_fn, regularizer=None):
    """
    Wrapper to switch an eager model into compiled JIT mode for maximum throughput.
    """
    return JITCompiler(model, optimizer, loss_fn, regularizer)