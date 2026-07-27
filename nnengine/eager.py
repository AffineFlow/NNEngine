import functools
from . import _backend

_THREAD_LOCAL_TAPE = None

def get_eager_tape() -> _backend.Tape:
    global _THREAD_LOCAL_TAPE
    if _THREAD_LOCAL_TAPE is None:
        _THREAD_LOCAL_TAPE = _backend.Tape(record_ops=True)
    return _THREAD_LOCAL_TAPE

def eager_step(optimizer, loss_fn, regularizer=None):
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(model, x_batch, y_batch, *args, **kwargs):
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
                
                # Apply the L2 penalty gradients if a regularizer is provided
                if regularizer is not None:
                    loss_val += regularizer.apply(model.parameters())
                    
            optimizer.step()
            return loss_val
        return wrapper
    return decorator