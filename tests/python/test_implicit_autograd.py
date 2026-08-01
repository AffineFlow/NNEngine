import pytest
import numpy as np
import nnengine as nne

def test_implicit_forward_and_backward():
    """Verify that basic operations are recorded to the global tape automatically."""
    np_a = np.array([2.0, 3.0], dtype=np.float32)
    np_b = np.array([4.0, 5.0], dtype=np.float32)
    
    a = nne.Tensor(np_a)
    a.requires_grad = True
    
    b = nne.Tensor(np_b)
    b.requires_grad = True
    
    # Implicitly records to the C++ global tape
    c = a * b
    out = c.sum()
    
    # Executes backprop and automatically destroys the graph
    out.backward()
    
    # d(out)/da = b
    np.testing.assert_allclose(np.array(a.grad), np_b, rtol=1e-5)
    # d(out)/db = a
    np.testing.assert_allclose(np.array(b.grad), np_a, rtol=1e-5)


def test_no_grad_context():
    """Verify that nne.no_grad() prevents the tape from tracking operations."""
    a = nne.Tensor(np.array([2.0, 3.0], dtype=np.float32))
    a.requires_grad = True
    
    # 1. Math inside no_grad should not track gradients
    with nne.no_grad():
        b = a * 3.0
    
    # Because 'b' wasn't tracked, its requires_grad should be false
    assert not b.requires_grad
    
    # Calling backward on an untracked tensor should throw an error
    with pytest.raises(RuntimeError):
        b.sum().backward()
        
    # 2. Math outside no_grad resumes tracking
    c = a * 4.0
    assert c.requires_grad
    c.sum().backward()
    
    np.testing.assert_allclose(np.array(a.grad), [4.0, 4.0])


def test_retain_graph():
    """Verify that retain_graph=True allows multiple backward passes on the same graph."""
    a = nne.Tensor(np.array([2.0, 3.0], dtype=np.float32))
    a.requires_grad = True
    
    b = a * 5.0
    out = b.sum()
    
    # First backward pass, keeping the graph intact
    out.backward(retain_graph=True)
    np.testing.assert_allclose(np.array(a.grad), [5.0, 5.0])
    
    # Second backward pass. Replays the retained graph. 
    # Note: Our zero-allocation engine accumulates intermediate gradients when retain_graph=True
    out.backward(retain_graph=False)
    np.testing.assert_allclose(np.array(a.grad), [15.0, 15.0])


def test_loss_backward_implicit_reset():
    """Verify that nne.Loss triggers graph destruction natively."""
    model = nne.DenseLayer(2, 1)
    
    x = nne.Tensor(np.array([[1.0, 2.0]], dtype=np.float32))
    y = nne.Tensor(np.array([[3.0]], dtype=np.float32))
    
    loss_fn = nne.MSELoss()
    
    # Epoch 1
    preds = model(x)
    loss_fn.forward(preds, y)
    loss_fn.backward() # Graph destroyed here
    
    # Epoch 2 (Would crash if the arena wasn't properly reset in Epoch 1)
    preds2 = model(x)
    loss_fn.forward(preds2, y)
    loss_fn.backward() 
    
    assert True