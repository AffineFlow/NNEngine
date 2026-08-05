import pytest
import numpy as np
import affineflow_nn as nne

def test_sgd_optimizer_step():
    """Verify that SGD correctly updates weights based on gradients."""
    nne.set_seed(42)
    layer = nne.DenseLayer(2, 2)
    
    w_initial = np.array(layer.parameters()[0].data).copy()
    layer.parameters()[0].grad[:] = 1.0
    
    # Correct initialization: optimizer takes lr, parameters are injected separately
    optimizer = nne.SGD(learning_rate=0.1)
    optimizer.set_parameters(layer.parameters())
    
    optimizer.step()
    w_expected = w_initial - (0.1 * 1.0)
    
    np.testing.assert_allclose(layer.parameters()[0].data, w_expected, rtol=1e-5)
    
    optimizer.zero_grad()
    np.testing.assert_allclose(layer.parameters()[0].grad, 0.0, rtol=1e-5)

def test_adam_optimizer_step():
    """Verify Adam optimizer applies updates."""
    layer = nne.DenseLayer(2, 2)
    w_initial = np.array(layer.parameters()[0].data).copy()
    layer.parameters()[0].grad[:] = 0.5
    
    optimizer = nne.Adam(learning_rate=0.01)
    optimizer.set_parameters(layer.parameters())
    optimizer.step()
    
    w_new = np.array(layer.parameters()[0].data)
    assert not np.allclose(w_initial, w_new), "Adam optimizer failed to update weights."

def test_adamw_optimizer_step():
    """Verify AdamW optimizer applies weight decay."""
    layer = nne.DenseLayer(2, 2)
    layer.parameters()[0].data[:] = 1.0
    layer.parameters()[0].grad[:] = 0.0
    
    optimizer = nne.AdamW(learning_rate=0.01, weight_decay=0.1)
    optimizer.set_parameters(layer.parameters())
    optimizer.step()
    
    w_new = np.array(layer.parameters()[0].data)
    np.testing.assert_allclose(w_new, 0.999, rtol=1e-4)