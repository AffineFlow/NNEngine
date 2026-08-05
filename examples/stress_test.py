import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader as TorchDataLoader, TensorDataset

import affineflow_nn as nne

device = torch.device("cpu")

# ==========================================
# INITIALIZATION
# ==========================================
def set_global_seeds(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    nne.set_seed(seed)

# ==========================================
# TASK 1: HEAVY REGRESSION (100k SAMPLES, WIDE MLP)
# ==========================================
class PyTorchHeavyMLP(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.net(x)

class AffineFlowNNHeavyMLP(nne.Module):
    def __init__(self, in_features):
        super().__init__()
        self.fc1 = nne.DenseLayer(in_features, 512)
        self.act1 = nne.ReLULayer()
        
        self.fc2 = nne.DenseLayer(512, 256)
        self.act2 = nne.ReLULayer()
        
        self.fc3 = nne.DenseLayer(256, 128)
        self.act3 = nne.ReLULayer()
        
        self.fc4 = nne.DenseLayer(128, 1)

    def forward(self, x):
        x = self.act1(self.fc1(x))
        x = self.act2(self.fc2(x))
        x = self.act3(self.fc3(x))
        return self.fc4(x)

def run_heavy_regression(seeds=[42]):
    print(f"\n{'-'*20} STRESS TEST: MASSIVE WIDE MLP {'-'*20}")
    
    # Generate 100k samples, 100 features in-memory
    n_samples, n_features = 100000, 100
    X_train_s = np.random.randn(n_samples, n_features).astype(np.float32)
    y_train = np.random.randn(n_samples, 1).astype(np.float32)

    epochs = 20
    batch_size = 256
    lr = 0.001

    # --- PyTorch ---
    print("Evaluating PyTorch (ATen)...")
    torch.manual_seed(seeds[0])
    pt_model = PyTorchHeavyMLP(n_features).to(device)
    pt_opt = optim.Adam(pt_model.parameters(), lr=lr)
    pt_loss_fn = nn.MSELoss()
    
    train_ds = TensorDataset(torch.tensor(X_train_s), torch.tensor(y_train))
    train_loader = TorchDataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)

    t0 = time.perf_counter()
    pt_model.train()
    for epoch in range(epochs):
        for bx, by in train_loader:
            pt_opt.zero_grad()
            loss = pt_loss_fn(pt_model(bx), by)
            loss.backward()
            pt_opt.step()
    pt_time = time.perf_counter() - t0

    # --- AffineFlow-NN JIT ---
    print("Evaluating AffineFlow-NN (JIT)...")
    nne.set_seed(seeds[0])
    nn_model = AffineFlowNNHeavyMLP(n_features)
    nn_opt = nne.Adam(learning_rate=lr)
    nn_loss = nne.MSELoss()
    
    trainer = nne.JITCompiler(nn_model, nn_opt, nn_loss)
    nne_loader = nne.DataLoader(X_train_s, y_train, batch_size=batch_size, shuffle=True, drop_last=True)

    t0 = time.perf_counter()
    trainer.fit(nne_loader, epochs=epochs, verbose=False)
    nn_time = time.perf_counter() - t0

    # --- AffineFlow-NN Eager ---
    print("Evaluating AffineFlow-NN (Eager)...")
    nne.set_seed(seeds[0])
    eag_model = AffineFlowNNHeavyMLP(n_features)
    eag_opt = nne.Adam(learning_rate=lr)
    eag_opt.set_parameters(eag_model.parameters())
    eag_loss = nne.MSELoss()
        
    eag_loader = nne.DataLoader(X_train_s, y_train, batch_size=batch_size, shuffle=True, drop_last=True)
    
    bx = nne.Tensor(np.zeros((batch_size, n_features), dtype=np.float32))
    by = nne.Tensor(np.zeros((batch_size, 1), dtype=np.float32))

    t0 = time.perf_counter()
    eag_model.train(True)
    for epoch in range(epochs):
        eag_loader.reset()
        while eag_loader.has_next():
            eag_loader.next_batch(bx, by)
            
            # New implicit UX
            eag_opt.zero_grad()
            preds = eag_model(bx)
            loss = eag_loss.forward(preds, by)
            eag_loss.backward()
            eag_opt.step()
            
    eag_time = time.perf_counter() - t0

    print("\n--- Heavy MLP Results ---")
    print(f"PyTorch      | Time: {pt_time:.4f}s")
    print(f"AffineFlow-NN JIT | Time: {nn_time:.4f}s")
    print(f"AffineFlow-NN Eag | Time: {eag_time:.4f}s")


# ==========================================
# TASK 2: HEAVY CNN (3 CHANNELS, 3 CONV LAYERS)
# ==========================================
class PyTorchHeavyCNN(nn.Module):
    def __init__(self, in_channels, in_h, in_w, num_classes):
        super().__init__()
        self.in_channels, self.in_h, self.in_w = in_channels, in_h, in_w
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=5, stride=2, padding=2)
        self.act1 = nn.LeakyReLU(0.01) 
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.act2 = nn.LeakyReLU(0.01) 
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.act3 = nn.LeakyReLU(0.01) 
        self.flatten = nn.Flatten()
        
        out_h = in_h // 8
        out_w = in_w // 8
        self.fc = nn.Linear(128 * out_h * out_w, num_classes)

    def forward(self, x):
        x = x.view(-1, self.in_channels, self.in_h, self.in_w)
        x = self.act1(self.conv1(x))
        x = self.act2(self.conv2(x))
        x = self.act3(self.conv3(x))
        x = self.flatten(x)
        return self.fc(x)

class AffineFlowNNHeavyCNN(nne.Module):
    def __init__(self, in_channels, in_h, in_w, num_classes):
        super().__init__()
        self.conv1 = nne.Conv2dLayer(in_channels, 32, in_h, in_w, kernel_size=5, stride=2, pad=2)
        self.act1 = nne.LeakyReLULayer(0.01) 
        
        out_h1, out_w1 = in_h // 2, in_w // 2
        self.conv2 = nne.Conv2dLayer(32, 64, out_h1, out_w1, kernel_size=3, stride=2, pad=1)
        self.act2 = nne.LeakyReLULayer(0.01) 
        
        out_h2, out_w2 = out_h1 // 2, out_w1 // 2
        self.conv3 = nne.Conv2dLayer(64, 128, out_h2, out_w2, kernel_size=3, stride=2, pad=1)
        self.act3 = nne.LeakyReLULayer(0.01)
        
        out_h3, out_w3 = out_h2 // 2, out_w2 // 2
        self.fc = nne.DenseLayer(128 * out_h3 * out_w3, num_classes)

    def forward(self, x):
        x = self.act1(self.conv1(x))
        x = self.act2(self.conv2(x))
        x = self.act3(self.conv3(x))
        return self.fc(x)

def run_heavy_cnn(seeds=[42]):
    print(f"\n{'-'*20} STRESS TEST: DEEP SPATIAL CNN {'-'*20}")
    
    # 2500 synthetic RGB images (3x64x64)
    n_samples, channels, img_h, img_w, num_classes = 2500, 3, 64, 64, 10
    X_train_cnn = np.random.rand(n_samples, channels, img_h, img_w).astype(np.float32)
    y_train = np.random.choice(num_classes, n_samples)
    y_train_onehot = np.eye(num_classes, dtype=np.float32)[y_train]
    
    epochs = 15
    batch_size = 32
    lr = 0.001

    # --- PyTorch ---
    print("Evaluating PyTorch (ATen)...")
    torch.manual_seed(seeds[0])
    pt_model = PyTorchHeavyCNN(channels, img_h, img_w, num_classes).to(device)
    pt_opt = optim.Adam(pt_model.parameters(), lr=lr)
    pt_loss_fn = nn.CrossEntropyLoss()
    
    train_ds = TensorDataset(torch.tensor(X_train_cnn), torch.tensor(y_train))
    train_loader = TorchDataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)

    t0 = time.perf_counter()
    pt_model.train()
    for epoch in range(epochs):
        for bx, by in train_loader:
            pt_opt.zero_grad()
            loss = pt_loss_fn(pt_model(bx), by)
            loss.backward()
            pt_opt.step()
    pt_time = time.perf_counter() - t0

    # --- AffineFlow-NN JIT ---
    print("Evaluating AffineFlow-NN (JIT)...")
    nne.set_seed(seeds[0])
    nn_model = AffineFlowNNHeavyCNN(channels, img_h, img_w, num_classes)
    nn_opt = nne.Adam(learning_rate=lr)
    nn_loss = nne.SoftmaxCrossEntropyLoss()
    
    trainer = nne.JITCompiler(nn_model, nn_opt, nn_loss)
    nne_loader = nne.DataLoader(X_train_cnn, y_train_onehot, batch_size=batch_size, shuffle=True, drop_last=True)

    t0 = time.perf_counter()
    trainer.fit(nne_loader, epochs=epochs, verbose=False)
    nn_time = time.perf_counter() - t0

    # --- AffineFlow-NN Eager ---
    print("Evaluating AffineFlow-NN (Eager)...")
    nne.set_seed(seeds[0])
    eag_model = AffineFlowNNHeavyCNN(channels, img_h, img_w, num_classes)
    eag_opt = nne.Adam(learning_rate=lr)
    eag_opt.set_parameters(eag_model.parameters())
    eag_loss = nne.SoftmaxCrossEntropyLoss()
        
    eag_loader = nne.DataLoader(X_train_cnn, y_train_onehot, batch_size=batch_size, shuffle=True, drop_last=True)
    bx = nne.Tensor(np.zeros((batch_size, channels, img_h, img_w), dtype=np.float32))
    by = nne.Tensor(np.zeros((batch_size, num_classes), dtype=np.float32))

    t0 = time.perf_counter()
    eag_model.train(True)
    for epoch in range(epochs):
        eag_loader.reset()
        while eag_loader.has_next():
            eag_loader.next_batch(bx, by)
            
            # New implicit UX
            eag_opt.zero_grad()
            preds = eag_model(bx)
            loss = eag_loss.forward(preds, by)
            eag_loss.backward()
            eag_opt.step()
            
    eag_time = time.perf_counter() - t0

    print("\n--- Heavy CNN Results ---")
    print(f"PyTorch      | Time: {pt_time:.4f}s")
    print(f"AffineFlow-NN JIT | Time: {nn_time:.4f}s")
    print(f"AffineFlow-NN Eag | Time: {eag_time:.4f}s")

if __name__ == "__main__":
    print("Starting Synthetic AffineFlow-NN Stress Tests...")
    run_heavy_regression()
    run_heavy_cnn()