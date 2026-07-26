import pytest
import numpy as np
import nnengine as nne

def test_tensor_arithmetic():
    np_a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    np_b = np.array([[2.0, 2.0], [2.0, 2.0]], dtype=np.float32)
    
    tape = nne.Tape(record_ops=True)
    with tape:
        a = tape.push_tensor(nne.Tensor(np_a), requires_grad=True)
        b = tape.push_tensor(nne.Tensor(np_b), requires_grad=True)
        
        # Test Math Operators
        c = a + b
        d = c * a
        e = d - b
        out = e / a
        
        out.grad[:] = 1.0
        tape.backward()
        
    np.testing.assert_allclose(np.array(out), ((np_a + np_b) * np_a - np_b) / np_a, rtol=1e-5)
    
    # Verify gradients fired correctly
    assert not np.allclose(np.array(a.grad), 0.0)

def test_tensor_matmul():
    np_a = np.array([[1.0, 2.0]], dtype=np.float32) # 1x2
    np_b = np.array([[3.0], [4.0]], dtype=np.float32) # 2x1
    
    tape = nne.Tape(record_ops=True)
    with tape:
        a = tape.push_tensor(nne.Tensor(np_a), requires_grad=True)
        b = tape.push_tensor(nne.Tensor(np_b), requires_grad=True)
        
        out = a @ b
        
        out.grad[:] = 1.0
        tape.backward()
        
    assert np.array(out)[0, 0] == 11.0
    np.testing.assert_array_equal(np.array(a.grad), np_b.T)
    np.testing.assert_array_equal(np.array(b.grad), np_a.T)