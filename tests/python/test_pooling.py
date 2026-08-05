import pytest
import numpy as np
import affineflow_nn as nne

def test_maxpool2d_forward_math():
    # 1 batch, 1 channel, 4x4 input
    X_np = np.array([
        [1,  2,  3,  4],
        [5,  6,  7,  8],
        [9,  10, 11, 12],
        [13, 14, 15, 16]
    ], dtype=np.float32).reshape(1, 1, 4, 4)
    
    layer = nne.MaxPool2dLayer(channels=1, in_h=4, in_w=4, kernel_size=2, stride=2, pad=0)
    
    tape = nne.Tape(record_ops=False)
    with tape:
        X_tensor = tape.push_tensor(nne.Tensor(X_np), requires_grad=False)
        out = layer(X_tensor)
        
    out_np = np.array(out)
    
    # AffineFlow-NN outputs flat channels per batch, so 2x2 output becomes flat length 4
    expected = np.array([
        [6, 8, 14, 16]
    ], dtype=np.float32)
    
    np.testing.assert_array_equal(out_np, expected)

def test_avgpool2d_forward_math():
    # 1 batch, 1 channel, 4x4 input
    X_np = np.array([
        [1,  2,  3,  4],
        [5,  6,  7,  8],
        [9,  10, 11, 12],
        [13, 14, 15, 16]
    ], dtype=np.float32).reshape(1, 1, 4, 4)
    
    layer = nne.AvgPool2dLayer(channels=1, in_h=4, in_w=4, kernel_size=2, stride=2, pad=0)
    
    tape = nne.Tape(record_ops=False)
    with tape:
        X_tensor = tape.push_tensor(nne.Tensor(X_np), requires_grad=False)
        out = layer(X_tensor)
        
    out_np = np.array(out)
    
    # Averages of the 4 2x2 blocks
    expected = np.array([
        [3.5, 5.5, 11.5, 13.5]
    ], dtype=np.float32)
    
    np.testing.assert_array_equal(out_np, expected)