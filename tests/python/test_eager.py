import pytest
import numpy as np
import affineflow_nn as nne

def test_dynamic_graph_control_flow():
    """Verify that Python 'if' statements can dictate the computational graph dynamically."""
    nne.set_seed(42)
    
    tape = nne.Tape(record_ops=True)
    X_np = np.array([[-1.0, 2.0], [3.0, -4.0]], dtype=np.float32)
    
    with tape:
        x_tensor = tape.push_tensor(nne.Tensor(X_np), requires_grad=True)
        
        # Dynamic graph execution
        if x_tensor.mean().data[0] < -10.0:
            out = x_tensor * -1.0
        else:
            out = x_tensor.relu()
            
        out.sum().backward()
        
    out_np = np.array(out)
    
    # Assert forward pass followed the `else` branch (mean is 0.0, which is > -10.0)
    assert out_np[0, 0] == 0.0
    assert out_np[0, 1] == 2.0
    assert out_np[1, 0] == 3.0
    
    # Assert the gradients correspond to the ReLU branch
    grad_np = np.array(x_tensor.grad)
    assert grad_np[0, 0] == 0.0
    assert grad_np[0, 1] == 1.0
    assert grad_np[1, 0] == 1.0


def test_eager_decorator_training():
    """Verify that the @nne.eager decorator effectively trains a model without crashing."""
    nne.set_seed(42)
    
    # XOR Dataset
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    Y = np.array([[0], [1], [1], [0]], dtype=np.float32)
    
    # Build Model
    model = nne.Module()
    model.l1 = nne.DenseLayer(2, 8)
    model.act = nne.ReLULayer()
    model.l2 = nne.DenseLayer(8, 1)
    
    optimizer = nne.Adam(learning_rate=0.1)
    optimizer.set_parameters(model.parameters())
    loss_fn = nne.MSELoss()
    
    # Decorate the step function for zero-allocation eager execution
    @nne.eager(optimizer=optimizer, loss_fn=loss_fn)
    def train_step(mod, x):  # <-- Removed 'y' here
        h = mod.act(mod.l1(x))
        return mod.l2(h)
        
    initial_loss = train_step(model, X, Y)
    
    for _ in range(50):
        final_loss = train_step(model, X, Y)
        
    assert final_loss < initial_loss, "Eager model failed to learn; loss did not decrease."
    assert final_loss < 0.1, f"Eager model failed to converge. Final loss: {final_loss}"


def test_tape_reset_safety():
    """Verify that calling tape.reset() manually in Python does not leak memory or crash."""
    tape = nne.Tape(record_ops=True)
    X = np.array([[1.0, 2.0]], dtype=np.float32)
    
    for _ in range(10):
        tape.reset()
        with tape:
            a = tape.push_tensor(nne.Tensor(X), requires_grad=True)
            b = a * 2.0
            c = b + 5.0
            c.sum().backward()
            
    # If the arena didn't handle destructors properly, this would have segfaulted.
    assert np.array(c)[0, 0] == 7.0
    assert np.array(a.grad)[0, 0] == 2.0

def test_eager_decorator_with_regularization():
    """Verify that the @nne.eager decorator correctly applies regularization without crashing."""
    nne.set_seed(42)
    
    X = np.array([[1.0, 2.0]], dtype=np.float32)
    Y = np.array([[1.0]], dtype=np.float32)
    
    # Simple model
    model = nne.DenseLayer(2, 1)
    
    optimizer = nne.SGD(learning_rate=0.01)
    optimizer.set_parameters(model.parameters())
    loss_fn = nne.MSELoss()
    reg = nne.L2Regularizer(l2=0.5)
    
    # Decorate with the regularizer attached
    @nne.eager(optimizer=optimizer, loss_fn=loss_fn, regularizer=reg)
    def train_step(mod, x):
        return mod(x)
        
    bx = nne.Tensor(X)
    by = nne.Tensor(Y)
    
    # Execute a step. This should now run flawlessly.
    loss = train_step(model, bx, by)
    
    # Ensure a valid Python float was returned back across the pybind11 boundary
    assert isinstance(loss, float)
    assert loss > 0.0, "Loss calculation with regularization failed."