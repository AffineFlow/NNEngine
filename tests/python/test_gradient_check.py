import pytest
import numpy as np
import nnengine as nne

def compute_numerical_gradient(forward_fn, X_np, weight_tensor, eps=1e-4):
    grad = np.zeros_like(weight_tensor.data)
    flat_weights = weight_tensor.data.ravel()
    flat_grad = grad.ravel()
    
    for i in range(len(flat_weights)):
        orig_val = flat_weights[i]
        
        flat_weights[i] = orig_val + eps
        out_plus = forward_fn(X_np)
        loss_plus = np.sum(out_plus) # out_plus is a pure numpy array now
        
        flat_weights[i] = orig_val - eps
        out_minus = forward_fn(X_np)
        loss_minus = np.sum(out_minus)
        
        flat_grad[i] = (loss_plus - loss_minus) / (2 * eps)
        flat_weights[i] = orig_val
        
    return grad

def test_dense_layer_gradient_accuracy():
    nne.set_seed(42)
    layer = nne.DenseLayer(4, 3)
    X_np = np.random.randn(2, 4).astype(np.float32)
    tape = nne.Tape(record_ops=True)
    
    with tape:
        X_tensor = tape.push_tensor(nne.Tensor(X_np), requires_grad=True)
        out_tensor = layer(X_tensor)
        out_tensor.grad[:] = 1.0
        tape.backward()

    analytical_grad_w = np.array(layer.parameters()[0].grad)

    def forward_pass(x_input):
        eval_tape = nne.Tape(record_ops=False)
        with eval_tape:
            out = layer(eval_tape.push_tensor(nne.Tensor(x_input), requires_grad=False))
            # Critical: Extract data to a pure NumPy copy before the tape dies
            return np.array(out, copy=True) 

    num_grad_w = compute_numerical_gradient(forward_pass, X_np, layer.parameters()[0])
    
    rel_error = np.linalg.norm(num_grad_w - analytical_grad_w) / (np.linalg.norm(num_grad_w + analytical_grad_w) + 1e-8)
    assert rel_error < 1e-3, f"DenseLayer gradient check failed: {rel_error}"

def test_conv2d_layer_gradient_accuracy():
    nne.set_seed(42)
    layer = nne.Conv2dLayer(2, 3, 5, 5, 3)
    X_np = np.random.randn(2, 2, 5, 5).astype(np.float32)
    tape = nne.Tape(record_ops=True)

    with tape:
        X_tensor = tape.push_tensor(nne.Tensor(X_np), requires_grad=True)
        out_tensor = layer(X_tensor)
        out_tensor.grad[:] = 1.0
        tape.backward()

    analytical_grad_w = np.array(layer.parameters()[0].grad)

    def forward_pass(x_input):
        eval_tape = nne.Tape(record_ops=False)
        with eval_tape:
            out = layer(eval_tape.push_tensor(nne.Tensor(x_input), requires_grad=False))
            # Critical: Extract data to a pure NumPy copy before the tape dies
            return np.array(out, copy=True)

    num_grad_w = compute_numerical_gradient(forward_pass, X_np, layer.parameters()[0])
    
    rel_error = np.linalg.norm(num_grad_w - analytical_grad_w) / (np.linalg.norm(num_grad_w + analytical_grad_w) + 1e-8)
    assert rel_error < 1e-3, f"Conv2d gradient check failed: {rel_error}"