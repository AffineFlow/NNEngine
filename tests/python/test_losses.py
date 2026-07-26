import pytest
import numpy as np
import nnengine as nne

def test_mse_loss():
    loss_fn = nne.MSELoss()
    tape = nne.Tape(record_ops=True)
    
    preds_np = np.array([[0.5, 0.2], [0.1, 0.9]], dtype=np.float32)
    targets_np = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    
    with tape:
        preds = tape.push_tensor(nne.Tensor(preds_np), requires_grad=True)
        targets = tape.push_tensor(nne.Tensor(targets_np), requires_grad=False)
        
        # 'loss' is a native Python float now
        loss = loss_fn.forward(preds, targets)
        
        # Trigger native loss derivative computation, then replay the tape
        loss_fn.backward()
        tape.backward()
        
    expected_loss = np.mean((preds_np - targets_np) ** 2)
    np.testing.assert_allclose(loss, expected_loss, rtol=1e-5)
    
    expected_grad = 2 * (preds_np - targets_np) / preds_np.size
    np.testing.assert_allclose(preds.grad, expected_grad, rtol=1e-5)

def test_softmax_cross_entropy():
    loss_fn = nne.SoftmaxCrossEntropyLoss()
    tape = nne.Tape(record_ops=True)
    
    logits_np = np.array([[2.0, 1.0, 0.1], [0.5, 2.5, 0.3]], dtype=np.float32)
    targets_np = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    
    with tape:
        logits = tape.push_tensor(nne.Tensor(logits_np), requires_grad=True)
        targets = tape.push_tensor(nne.Tensor(targets_np), requires_grad=False)
        
        loss = loss_fn.forward(logits, targets)
        loss_fn.backward()
        tape.backward()

    exp_logits = np.exp(logits_np - np.max(logits_np, axis=1, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    
    expected_grad = (probs - targets_np) / logits_np.shape[0]
    np.testing.assert_allclose(logits.grad, expected_grad, rtol=1e-5, atol=1e-5)