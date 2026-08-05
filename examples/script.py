import os
import time
import warnings
import numpy as np

# Suppress Scikit-Learn convergence warnings for a clean CLI output
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader as TorchDataLoader, TensorDataset

device = torch.device("cpu")

from sklearn.datasets import fetch_california_housing, fetch_olivetti_faces
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

import affineflow_nn as nne

# ==========================================
# INITIALIZATION
# ==========================================
def set_global_seeds(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    nne.set_seed(seed)

# ==========================================
# TASK 1: DEEP REGRESSION (CALIFORNIA HOUSING)
# ==========================================
class PyTorchDeepMLP(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)

class AffineFlowNNDeepMLP(nne.Module):
    def __init__(self, in_features):
        super().__init__()
        self.fc1 = nne.DenseLayer(in_features, 128)
        self.act1 = nne.ReLULayer()
        
        self.fc2 = nne.DenseLayer(128, 64)
        self.act2 = nne.ReLULayer()
        
        self.fc3 = nne.DenseLayer(64, 1)

    def forward(self, x):
        x = self.act1(self.fc1(x))
        x = self.act2(self.fc2(x))
        return self.fc3(x)

def run_regression_benchmark(seeds=[42, 1337, 2026]):
    print(f"\n{'-'*20} BENCHMARK: DEEP REGRESSION (CALIFORNIA HOUSING) {'-'*20}")
    
    X, y = fetch_california_housing(return_X_y=True)
    y = y.reshape(-1, 1)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X.astype(np.float32), y.astype(np.float32), test_size=0.2, random_state=42
    )
    
    scaler_x = StandardScaler()
    X_train_s = scaler_x.fit_transform(X_train).astype(np.float32)
    X_test_s = scaler_x.transform(X_test).astype(np.float32)

    epochs = 50
    batch_size = 128
    lr = 0.01
    weight_decay = 1e-4

    # --- 1. Scikit-Learn ---
    print("Evaluating Scikit-Learn MLPRegressor across seeds...")
    best_sk_mse = float("inf")
    best_sk_results = None

    for seed in seeds:
        sk_model = MLPRegressor(
            hidden_layer_sizes=(128, 64), activation='relu', solver='adam',
            alpha=weight_decay, batch_size=batch_size, learning_rate_init=lr,
            max_iter=epochs, random_state=seed, early_stopping=False
        )
        t0 = time.perf_counter()
        sk_model.fit(X_train_s, y_train.ravel())
        sk_time = time.perf_counter() - t0
        sk_preds = sk_model.predict(X_test_s)
        sk_mse = mean_squared_error(y_test, sk_preds)
        sk_r2 = r2_score(y_test, sk_preds)

        if sk_mse < best_sk_mse:
            best_sk_mse = sk_mse
            best_sk_results = (sk_mse, sk_r2, sk_time)
    sk_mse, sk_r2, sk_time = best_sk_results

    # --- 2. PyTorch ---
    print("Evaluating PyTorch (ATen) across seeds...")
    best_pt_mse = float("inf")
    best_pt_results = None

    for seed in seeds:
        torch.manual_seed(seed)
        pt_model = PyTorchDeepMLP(X_train.shape[1]).to(device)
        pt_opt = optim.Adam(pt_model.parameters(), lr=lr, weight_decay=weight_decay)
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
        
        pt_model.eval()
        with torch.no_grad():
            pt_preds = pt_model(torch.tensor(X_test_s)).numpy()
        pt_mse = mean_squared_error(y_test, pt_preds)
        pt_r2 = r2_score(y_test, pt_preds)

        if pt_mse < best_pt_mse:
            best_pt_mse = pt_mse
            best_pt_results = (pt_mse, pt_r2, pt_time)
    pt_mse, pt_r2, pt_time = best_pt_results

    # --- 3. AffineFlow-NN JIT ---
    print("Evaluating AffineFlow-NN (JIT) across seeds...")
    best_nn_mse = float("inf")
    best_nn_results = None

    for seed in seeds:
        nne.set_seed(seed)
        nn_model = AffineFlowNNDeepMLP(X_train.shape[1])
        nn_opt = nne.Adam(learning_rate=lr)
        nn_loss = nne.MSELoss()
        nn_reg = nne.L2Regularizer(l2=weight_decay)
        
        trainer = nne.JITCompiler(nn_model, nn_opt, nn_loss, regularizer=nn_reg)
        nne_loader = nne.DataLoader(X_train_s, y_train, batch_size=batch_size, shuffle=True, drop_last=True)

        t0 = time.perf_counter()
        trainer.fit(nne_loader, epochs=epochs, verbose=False)
        nn_time = time.perf_counter() - t0
        
        nn_preds = np.array(nn_model.predict(X_test_s))
        nn_mse = mean_squared_error(y_test, nn_preds)
        nn_r2 = r2_score(y_test, nn_preds)

        if nn_mse < best_nn_mse:
            best_nn_mse = nn_mse
            best_nn_results = (nn_mse, nn_r2, nn_time)
    nn_mse, nn_r2, nn_time = best_nn_results

    # --- 4. AffineFlow-NN Eager (PyTorch-like Implicit Execution) ---
    print("Evaluating AffineFlow-NN (Eager) across seeds...")
    best_eag_mse = float("inf")
    best_eag_results = None

    for seed in seeds:
        nne.set_seed(seed)
        eag_model = AffineFlowNNDeepMLP(X_train.shape[1])
        eag_opt = nne.Adam(learning_rate=lr)
        eag_opt.set_parameters(eag_model.parameters())
        eag_loss = nne.MSELoss()
        eag_reg = nne.L2Regularizer(l2=weight_decay)
            
        eag_loader = nne.DataLoader(X_train_s, y_train, batch_size=batch_size, shuffle=True, drop_last=True)
        
        bx = nne.Tensor(np.zeros((batch_size, X_train.shape[1]), dtype=np.float32))
        by = nne.Tensor(np.zeros((batch_size, 1), dtype=np.float32))

        t0 = time.perf_counter()
        eag_model.train(True)
        for epoch in range(epochs):
            eag_loader.reset()
            while eag_loader.has_next():
                eag_loader.next_batch(bx, by)
                
                # 1. Zero out previous gradients
                eag_opt.zero_grad()
                
                # 2. Forward pass implicitly constructs the graph natively
                preds = eag_model(bx)
                loss = eag_loss.forward(preds, by)
                
                # 3. Triggers C++ backprop and auto-resets the internal arena
                eag_loss.backward()
                
                # Apply regularization natively
                eag_reg.apply(eag_model.parameters())
                
                # 4. Update parameters
                eag_opt.step()
                
        eag_time = time.perf_counter() - t0
        
        # predict() inherently uses no_grad natively
        eag_preds = np.array(eag_model.predict(X_test_s))
        eag_mse = mean_squared_error(y_test, eag_preds)
        eag_r2 = r2_score(y_test, eag_preds)

        if eag_mse < best_eag_mse:
            best_eag_mse = eag_mse
            best_eag_results = (eag_mse, eag_r2, eag_time)
    eag_mse, eag_r2, eag_time = best_eag_results

    # Results
    print("\n--- Regression Results (Best over Seeds) ---")
    print(f"Scikit-Learn | MSE: {sk_mse:.4f} | R2: {sk_r2:.4f} | Time: {sk_time:.4f}s")
    print(f"PyTorch      | MSE: {pt_mse:.4f} | R2: {pt_r2:.4f} | Time: {pt_time:.4f}s")
    print(f"AffineFlow-NN JIT | MSE: {nn_mse:.4f} | R2: {nn_r2:.4f} | Time: {nn_time:.4f}s")
    print(f"AffineFlow-NN Eag | MSE: {eag_mse:.4f} | R2: {eag_r2:.4f} | Time: {eag_time:.4f}s")


# ==========================================
# TASK 2: DEEP SPATIAL CLASSIFICATION
# ==========================================
class PyTorchDeepCNN(nn.Module):
    def __init__(self, in_h, in_w, num_classes):
        super().__init__()
        self.in_h, self.in_w = in_h, in_w

        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2)
        self.act1 = nn.LeakyReLU(0.01) 
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.act2 = nn.LeakyReLU(0.01) 
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(32 * (in_h // 4) * (in_w // 4), num_classes)

    def forward(self, x):
        x = x.view(-1, 1, self.in_h, self.in_w)
        x = self.pool1(self.act1(self.conv1(x)))
        x = self.pool2(self.act2(self.conv2(x)))
        x = self.flatten(x)
        return self.fc(x)

class AffineFlowNNDeepCNN(nne.Module):
    def __init__(self, in_h, in_w, num_classes):
        super().__init__()
        self.conv1 = nne.Conv2dLayer(1, 16, in_h, in_w, kernel_size=5, stride=1, pad=2)
        self.act1 = nne.LeakyReLULayer(0.01) 
        self.pool1 = nne.MaxPool2dLayer(16, in_h, in_w, kernel_size=2, stride=2, pad=0)
        
        out_h1, out_w1 = in_h // 2, in_w // 2
        self.conv2 = nne.Conv2dLayer(16, 32, out_h1, out_w1, kernel_size=3, stride=1, pad=1)
        self.act2 = nne.LeakyReLULayer(0.01) 
        self.pool2 = nne.MaxPool2dLayer(32, out_h1, out_w1, kernel_size=2, stride=2, pad=0)
        
        out_h2, out_w2 = out_h1 // 2, out_w1 // 2
        
        self.flatten = nne.FlattenLayer()
        self.fc = nne.DenseLayer(32 * out_h2 * out_w2, num_classes)

    def forward(self, x):
        x = self.pool1(self.act1(self.conv1(x)))
        x = self.pool2(self.act2(self.conv2(x)))
        x = self.flatten(x)
        return self.fc(x)

def run_classification_benchmark(seeds=[42, 1337, 2026]):
    print(f"\n{'-'*20} BENCHMARK: DEEP CNN (OLIVETTI FACES) {'-'*20}")
    
    faces = fetch_olivetti_faces()
    X, y = faces.data, faces.target
    img_h, img_w = 64, 64
    num_classes = 40
    
    X_train, X_test, y_train, y_test = train_test_split(
        X.astype(np.float32), y.astype(np.int64), test_size=0.2, random_state=42
    )
    
    X_train_cnn = X_train.reshape(-1, 1, img_h, img_w)
    X_test_cnn = X_test.reshape(-1, 1, img_h, img_w)
    y_train_onehot = np.eye(num_classes, dtype=np.float32)[y_train]
    
    epochs = 40
    batch_size = 32
    lr = 0.001

    # --- 1. PyTorch ---
    print("Evaluating PyTorch (ATen) across seeds...")
    best_pt_acc = -1.0
    best_pt_results = None

    for seed in seeds:
        torch.manual_seed(seed)
        pt_model = PyTorchDeepCNN(img_h, img_w, num_classes).to(device)
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
        
        pt_model.eval()
        with torch.no_grad():
            pt_preds = torch.argmax(pt_model(torch.tensor(X_test_cnn)), dim=1).numpy()
        pt_acc = accuracy_score(y_test, pt_preds) * 100

        if pt_acc > best_pt_acc:
            best_pt_acc = pt_acc
            best_pt_results = (pt_acc, pt_time)
    pt_acc, pt_time = best_pt_results

    # --- 2. AffineFlow-NN JIT ---
    print("Evaluating AffineFlow-NN (JIT) across seeds...")
    best_nn_acc = -1.0
    best_nn_results = None

    for seed in seeds:
        nne.set_seed(seed)
        nn_model = AffineFlowNNDeepCNN(img_h, img_w, num_classes)
        nn_opt = nne.Adam(learning_rate=lr)
        nn_loss = nne.SoftmaxCrossEntropyLoss()
        
        trainer = nne.JITCompiler(nn_model, nn_opt, nn_loss)
        nne_loader = nne.DataLoader(X_train_cnn, y_train_onehot, batch_size=batch_size, shuffle=True, drop_last=True)

        t0 = time.perf_counter()
        trainer.fit(nne_loader, epochs=epochs, verbose=False)
        nn_time = time.perf_counter() - t0
        
        nn_preds = np.argmax(np.array(nn_model.predict(X_test_cnn)), axis=1)
        nn_acc = accuracy_score(y_test, nn_preds) * 100

        if nn_acc > best_nn_acc:
            best_nn_acc = nn_acc
            best_nn_results = (nn_acc, nn_time)
    nn_acc, nn_time = best_nn_results

    # --- 3. AffineFlow-NN Eager (PyTorch-like Implicit Execution) ---
    print("Evaluating AffineFlow-NN (Eager) across seeds...")
    best_eag_acc = -1.0
    best_eag_results = None

    for seed in seeds:
        nne.set_seed(seed)
        eag_model = AffineFlowNNDeepCNN(img_h, img_w, num_classes)
        eag_opt = nne.Adam(learning_rate=lr)
        eag_opt.set_parameters(eag_model.parameters())
        eag_loss = nne.SoftmaxCrossEntropyLoss()
        
        eag_loader = nne.DataLoader(X_train_cnn, y_train_onehot, batch_size=batch_size, shuffle=True, drop_last=True)
        
        bx = nne.Tensor(np.zeros((batch_size, 1, img_h, img_w), dtype=np.float32))
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
        
        eag_preds = np.argmax(np.array(eag_model.predict(X_test_cnn)), axis=1)
        eag_acc = accuracy_score(y_test, eag_preds) * 100

        if eag_acc > best_eag_acc:
            best_eag_acc = eag_acc
            best_eag_results = (eag_acc, eag_time)
    eag_acc, eag_time = best_eag_results

    # Results
    print("\n--- Classification Results (Best over Seeds) ---")
    print(f"PyTorch      | Acc: {pt_acc:.2f}% | Time: {pt_time:.4f}s")
    print(f"AffineFlow-NN JIT | Acc: {nn_acc:.2f}% | Time: {nn_time:.4f}s")
    print(f"AffineFlow-NN Eag | Acc: {eag_acc:.2f}% | Time: {eag_time:.4f}s")

if __name__ == "__main__":
    print("Starting Comprehensive Multi-Seed AffineFlow-NN Validation Suite...")
    run_regression_benchmark()
    run_classification_benchmark()