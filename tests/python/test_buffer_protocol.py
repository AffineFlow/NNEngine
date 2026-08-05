import pytest
import numpy as np
import affineflow_nn as nne

def test_tensor_creation_from_numpy():
    """Verify that native C++ Tensors correctly inherit NumPy arrays."""
    np_arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    tensor = nne.Tensor(np_arr)
    
    assert tensor.shape == [2, 2], "Shape mismatch across pybind11 boundary"
    assert tensor.requires_grad == False, "Default numpy conversion should not require grad"

def test_numpy_zero_copy_buffer():
    """Verify the bi-directional buffer protocol."""
    np_arr = np.random.randn(5, 5).astype(np.float32)
    tensor = nne.Tensor(np_arr)
    
    # Extract back to NumPy via buffer protocol (no .data attribute needed)
    recovered_arr = np.array(tensor, copy=False)
    
    # Assert exact match
    np.testing.assert_array_equal(np_arr, recovered_arr)
    
    # Modify NumPy array and verify C++ Tensor sees it (Zero-copy verification)
    recovered_arr[0, 0] = 99.9
    assert np.array(tensor)[0, 0] == 99.9, "Buffer protocol is copying memory instead of sharing it!"

def test_tensor_properties():
    """Verify read/write access to native Tensor properties."""
    tensor = nne.Tensor(np.zeros((3, 3), dtype=np.float32))
    
    tensor.requires_grad = True
    assert tensor.requires_grad is True
    
    # Test shape resizing from Python
    tensor.shape = [9]
    assert tensor.shape == [9]
    assert np.array(tensor).shape == (9,)