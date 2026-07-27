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

def test_scalar_arithmetic_and_broadcasting():
    np_a = np.array([1.0, 2.0, 4.0], dtype=np.float32) # Adjusted 3.0 to 4.0 to avoid div-by-zero warning
    
    tape = nne.Tape(record_ops=True)
    with tape:
        a = tape.push_tensor(nne.Tensor(np_a), requires_grad=True)
        
        b = a + 2.0       # Forward add
        c = 5.0 - b       # Reverse sub
        d = c * 3.0       # Forward mul
        e = 12.0 / d      # Reverse div
        
        e.sum().backward() # Chain reduction + backprop!
        
    np.testing.assert_allclose(np.array(e), 12.0 / ((5.0 - (np_a + 2.0)) * 3.0), rtol=1e-5)
    # The gradient of 12 / ((5 - (a + 2)) * 3) is 4 / (a - 3)^2
    np.testing.assert_allclose(np.array(a.grad), 4.0 / ((np_a - 3.0) ** 2), rtol=1e-5)

def test_reductions():
    np_a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    
    tape = nne.Tape(record_ops=True)
    with tape:
        a = tape.push_tensor(nne.Tensor(np_a), requires_grad=True)
        mean_val = a.mean()
        mean_val.backward()
        
    assert np.array(mean_val)[0] == 2.5
    np.testing.assert_allclose(np.array(a.grad), 0.25, rtol=1e-5)

def test_inplace_updates():
    tensor = nne.Tensor(np.array([1.0, 2.0], dtype=np.float32))
    other = nne.Tensor(np.array([3.0, 4.0], dtype=np.float32))
    
    tensor += other
    np.testing.assert_array_equal(np.array(tensor), [4.0, 6.0])
    
    tensor -= 2.0
    np.testing.assert_array_equal(np.array(tensor), [2.0, 4.0])
    
    tensor *= 3.0
    np.testing.assert_array_equal(np.array(tensor), [6.0, 12.0])

def test_unary_operations():
    np_a = np.array([-1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    
    # Test ReLU
    tape = nne.Tape(record_ops=True)
    with tape:
        a = tape.push_tensor(nne.Tensor(np_a), requires_grad=True)
        out_relu = a.relu()
        out_relu.sum().backward()
        
    np.testing.assert_array_equal(np.array(out_relu), [0.0, 0.0, 1.0, 2.0])
    np.testing.assert_array_equal(np.array(a.grad), [0.0, 0.0, 1.0, 1.0])
    
    # Test Exp
    tape.reset()
    with tape:
        a = tape.push_tensor(nne.Tensor(np_a), requires_grad=True)
        out_exp = a.exp()
        out_exp.sum().backward()
        
    np.testing.assert_allclose(np.array(out_exp), np.exp(np_a), rtol=1e-5)
    np.testing.assert_allclose(np.array(a.grad), np.exp(np_a), rtol=1e-5)
    
    # Test Log (use strictly positive values to avoid NaN/Inf)
    np_pos = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    tape.reset()
    with tape:
        a = tape.push_tensor(nne.Tensor(np_pos), requires_grad=True)
        out_log = a.log()
        out_log.sum().backward()
        
    np.testing.assert_allclose(np.array(out_log), np.log(np_pos), rtol=1e-5)
    np.testing.assert_allclose(np.array(a.grad), 1.0 / np_pos, rtol=1e-5)

def test_transpose():
    np_a = np.array([[1.0, 2.0, 3.0], 
                     [4.0, 5.0, 6.0]], dtype=np.float32) # 2x3
                     
    np_mask = np.array([[1.0, 2.0], 
                        [3.0, 4.0], 
                        [5.0, 6.0]], dtype=np.float32) # 3x2
    
    tape = nne.Tape(record_ops=True)
    with tape:
        a = tape.push_tensor(nne.Tensor(np_a), requires_grad=True)
        mask = tape.push_tensor(nne.Tensor(np_mask), requires_grad=False)
        
        # Test .T property
        a_t = a.T
        
        # Multiply by the mask to ensure gradients have unique values depending on position
        out = a_t * mask
        out.sum().backward()
        
    # Forward check
    # Reshaping is necessary depending on how the Buffer Protocol exposes the multidimensional array to NumPy
    np.testing.assert_array_equal(np.array(a_t).reshape(3, 2), np_a.T)
    
    # Gradient check:
    # d(out) / d(a_t) = mask
    # d(out) / d(a) = mask.T
    expected_grad = np_mask.T
    np.testing.assert_array_equal(np.array(a.grad).reshape(2, 3), expected_grad)

def test_math_outside_tape_eager_fallback():
    """Verify that math operations outside a tape context evaluate correctly without crashing."""
    np_a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    np_b = np.array([[2.0, 0.5], [1.0, 2.0]], dtype=np.float32)
    
    a = nne.Tensor(np_a)
    b = nne.Tensor(np_b)
    
    # 1. Tensor-Tensor Arithmetic
    out_add = a + b
    np.testing.assert_allclose(np.array(out_add), np_a + np_b, err_msg="Eager Add failed")
    assert not out_add.requires_grad, "Eager fallback should not require gradients"
    
    out_sub = a - b
    np.testing.assert_allclose(np.array(out_sub), np_a - np_b, err_msg="Eager Sub failed")
    
    out_mul = a * b
    np.testing.assert_allclose(np.array(out_mul), np_a * np_b, err_msg="Eager Mul failed")
    
    out_div = a / b
    np.testing.assert_allclose(np.array(out_div), np_a / np_b, err_msg="Eager Div failed")
    
    out_matmul = a @ b
    np.testing.assert_allclose(np.array(out_matmul), np_a @ np_b, err_msg="Eager MatMul failed")
    
    # 2. Tensor-Scalar Arithmetic (Testing normal and reverse/R-ops)
    out_add_scalar = a + 2.0
    np.testing.assert_allclose(np.array(out_add_scalar), np_a + 2.0)
    
    out_rsub_scalar = 5.0 - a
    np.testing.assert_allclose(np.array(out_rsub_scalar), 5.0 - np_a)
    
    out_mul_scalar = a * 3.0
    np.testing.assert_allclose(np.array(out_mul_scalar), np_a * 3.0)
    
    out_rdiv_scalar = 12.0 / a
    np.testing.assert_allclose(np.array(out_rdiv_scalar), 12.0 / np_a)
    
    # 3. Unary Operations & Reductions
    np_pos = np.array([-1.0, 2.0, 4.0], dtype=np.float32)
    pos = nne.Tensor(np_pos)
    
    np.testing.assert_allclose(np.array(pos.relu()), np.maximum(0, np_pos))
    np.testing.assert_allclose(np.array(pos.exp()), np.exp(np_pos), rtol=1e-5)
    
    # Ensure log uses absolute positive values for the test to avoid NaNs
    np_strictly_pos = np.array([1.0, 2.0, 4.0], dtype=np.float32)
    strictly_pos = nne.Tensor(np_strictly_pos)
    np.testing.assert_allclose(np.array(strictly_pos.log()), np.log(np_strictly_pos), rtol=1e-5)
    
    np.testing.assert_allclose(np.array(pos.sum()), np.sum(np_pos))
    np.testing.assert_allclose(np.array(pos.mean()), np.mean(np_pos))
    
    # 4. Transposition
    # Because of the buffer protocol and flat memory, transpose outputs need a shape cast for numpy comparison
    out_t = a.T
    np.testing.assert_array_equal(np.array(out_t).reshape(2, 2), np_a.T)
    
    out_transpose = a.transpose()
    np.testing.assert_array_equal(np.array(out_transpose).reshape(2, 2), np_a.T)

def test_math_outside_tape_no_leakage():
    """Verify that multiple successive eager ops are safely memory-managed by Python."""
    np_start = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    t = nne.Tensor(np_start)
    
    # Chain several eager operations. If memory ownership is broken, this will segfault.
    t_chained = ((t * 2.0) + 1.5).relu().exp()
    
    expected = np.exp(np.maximum(0, (np_start * 2.0) + 1.5))
    np.testing.assert_allclose(np.array(t_chained), expected, rtol=1e-5)