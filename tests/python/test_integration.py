import pytest
import numpy as np
import affineflow_nn as nne

def test_mlp_training_loop():
    nne.set_seed(42)
    
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    Y = np.array([[0], [1], [1], [0]], dtype=np.float32)
    
    l1 = nne.DenseLayer(2, 8)
    act = nne.ReLULayer()
    l2 = nne.DenseLayer(8, 1)
    
    optimizer = nne.Adam(learning_rate=0.1)
    optimizer.set_parameters(l1.parameters() + l2.parameters())
    loss_fn = nne.MSELoss()
    
    losses = []
    
    for epoch in range(200):
        tape = nne.Tape(record_ops=True)
        
        with tape:
            x_tensor = tape.push_tensor(nne.Tensor(X), requires_grad=False)
            y_tensor = tape.push_tensor(nne.Tensor(Y), requires_grad=False)
            
            h = act(l1(x_tensor))
            out = l2(h)
            
            loss = loss_fn.forward(out, y_tensor)
            
            loss_fn.backward()
            tape.backward()
            
        optimizer.step()
        optimizer.zero_grad()
        
        losses.append(loss)
        
    assert losses[-1] < losses[0], "Model failed to learn; loss did not decrease."
    assert losses[-1] < 0.1, f"Model failed to converge. Final loss: {losses[-1]}"