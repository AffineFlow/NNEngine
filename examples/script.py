import time
import os
import numpy as np

# Force CPU threads for OpenBLAS and OpenMP (NNEngine's backend)
cores = str(os.cpu_count())
os.environ["OPENBLAS_NUM_THREADS"] = cores
os.environ["OMP_NUM_THREADS"] = cores

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Explicitly force PyTorch to use the CPU for a fair 1:1 hardware comparison
torch.set_num_threads(os.cpu_count())
device = torch.device("cpu")

from sklearn.datasets import load_digits, load_iris, fetch_olivetti_faces
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import nnengine as nne

# ==========================================
# Architectures (Parameterized for 2D Shapes)
# ==========================================
class PyTorchCNN(nn.Module):
    def __init__(self, in_h, in_w, num_classes):
        super().__init__()
        self.in_h = in_h
        self.in_w = in_w
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(16 * in_h * in_w, num_classes)

    def forward(self, x):
        x = x.view(-1, 1, self.in_h, self.in_w)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.flatten(x)
        return self.fc(x)

class NNEngineCNN(nne.Module):
    def __init__(self, in_h, in_w, num_classes):
        super().__init__()
        self.conv1 = nne.Conv2dLayer(
            in_channels=1, out_channels=16, 
            in_h=in_h, in_w=in_w, 
            kernel_size=3, stride=1, pad=1
        )
        self.relu = nne.ReLULayer()
        self.fc = nne.DenseLayer(16 * in_h * in_w, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        return self.fc(x)

# ==========================================
# Testing Logic
# ==========================================
def test_dataset_vs_pytorch(name, X, y, epochs, lr, batch_size, is_cnn=False, img_h=None, img_w=None):
    print(f"\n{'-'*10} {name} (NNEngine vs PyTorch) {'-'*10}")
    
    num_classes = len(np.unique(y))
    
    # 1. Data Prep
    X_train, X_test, y_train, y_test = train_test_split(
        X.astype(np.float32), y.astype(np.int64), test_size=0.2, random_state=42
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train).astype(np.float32)
    X_test_s = scaler.transform(X_test).astype(np.float32)

    y_train_onehot = np.eye(num_classes, dtype=np.float32)[y_train]

    # PyTorch DataLoaders (Tensors strictly mapped to CPU)
    train_ds = TensorDataset(
        torch.tensor(X_train_s).to(device), 
        torch.tensor(y_train).to(device)
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    
    # ---------------------------------------------------------
    # 2. PyTorch Training
    # ---------------------------------------------------------
    if is_cnn:
        pt_model = PyTorchCNN(img_h, img_w, num_classes).to(device)
    else:
        pt_model = nn.Sequential(
            nn.Linear(X.shape[1], 32), 
            nn.ReLU(), 
            nn.Linear(32, num_classes)
        ).to(device)
        
    pt_opt = optim.Adam(pt_model.parameters(), lr=lr)
    pt_loss_fn = nn.CrossEntropyLoss()
    pt_scheduler = optim.lr_scheduler.StepLR(pt_opt, step_size=20, gamma=0.5)

    t0 = time.perf_counter()
    pt_model.train()
    for epoch in range(epochs):
        for bx, by in train_loader:
            pt_opt.zero_grad()
            loss = pt_loss_fn(pt_model(bx), by)
            loss.backward()
            pt_opt.step()
        pt_scheduler.step()
    pt_time = time.perf_counter() - t0
    
    pt_model.eval()
    pt_preds = torch.argmax(pt_model(torch.tensor(X_test_s).to(device)), dim=1).cpu().numpy()
    pt_acc = accuracy_score(y_test, pt_preds) * 100

    # ---------------------------------------------------------
    # 3. NNEngine Training
    # ---------------------------------------------------------
    if is_cnn:
        nn_model = NNEngineCNN(img_h, img_w, num_classes)
    else: 
        class NNE_MLP(nne.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nne.DenseLayer(X.shape[1], 32)
                self.relu = nne.ReLULayer()
                self.fc2 = nne.DenseLayer(32, num_classes)
            def forward(self, x):
                return self.fc2(self.relu(self.fc1(x)))
        nn_model = NNE_MLP()

    nn_opt = nne.Adam(learning_rate=lr)
    nn_loss = nne.SoftmaxCrossEntropyLoss()
    nn_scheduler = nne.StepLR(nn_opt, step_size=20, gamma=0.5)
    
    trainer = nne.JITCompiler(nn_model, nn_opt, nn_loss)
    trainer._cpp_engine.set_scheduler(nn_scheduler)

    # Completely Native C++ Dataloader & Training Loop
    nne_loader = nne.DataLoader(X_train_s, y_train_onehot, batch_size=batch_size, shuffle=True, drop_last=True)

    t0 = time.perf_counter()
    trainer.fit(nne_loader, epochs=epochs, verbose=False)
    nn_time = time.perf_counter() - t0

    nn_preds = np.argmax(nn_model.predict(X_test_s), axis=1)
    nn_acc = accuracy_score(y_test, nn_preds) * 100

    # ---------------------------------------------------------
    # 4. Results
    # ---------------------------------------------------------
    print(f"🚀 NNEngine (JIT)   — Acc: {nn_acc:.2f}%  |  Time: {nn_time:.4f}s")
    print(f"🔥 PyTorch (ATen)   — Acc: {pt_acc:.2f}%  |  Time: {pt_time:.4f}s")
    if nn_time < pt_time:
        print(f"⚡ Speedup          — {pt_time / nn_time:.2f}x Faster")
    else:
        print(f"🐢 Slowdown         — {nn_time / pt_time:.2f}x Slower")

if __name__ == "__main__":
    print("⚔️  NNEngine vs PyTorch\n")
    
    # Test 1: Iris (MLP)
    iris = load_iris()
    test_dataset_vs_pytorch("Iris (MLP)", iris.data, iris.target, epochs=100, lr=0.01, batch_size=16)

    # Test 2: Handwritten Digits (CNN)
    digits = load_digits()
    test_dataset_vs_pytorch(
        "Digits (Conv2D CNN)", 
        digits.data, digits.target, 
        epochs=40, lr=0.005, batch_size=32, 
        is_cnn=True, img_h=8, img_w=8
    )

    # Test 3: Olivetti Faces (CNN)
    faces = fetch_olivetti_faces()
    test_dataset_vs_pytorch(
        "Olivetti Faces (Conv2D CNN)", 
        faces.data, faces.target, 
        epochs=40, lr=0.005, batch_size=32, 
        is_cnn=True, img_h=64, img_w=64
    )