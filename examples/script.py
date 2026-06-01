import argparse
import time
import os
import warnings
import numpy as np

cores = str(os.cpu_count())
os.environ["OPENBLAS_NUM_THREADS"] = cores
os.environ["OMP_NUM_THREADS"] = cores
os.environ["MKL_NUM_THREADS"] = cores

from sklearn.datasets import fetch_olivetti_faces, load_digits, load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.exceptions import ConvergenceWarning

import nnengine as nn
warnings.filterwarnings("ignore", category=ConvergenceWarning)

def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    nn.set_seed(seed)

class BenchmarkNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.DenseLayer(input_dim, hidden_dim)
        self.relu = nn.ReLULayer()
        self.drop = nn.DropoutLayer(p=0.2) 
        self.fc2 = nn.DenseLayer(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.drop(x)
        return self.fc2(x)


def test_dataset(name, X, y, epochs, lr, batch_size, hidden_size, seed, use_l2=True):
    print(f"\n{'-'*10} Testing {name} {'-'*10}")
    seed_everything(seed)

    num_features = X.shape[1]
    num_classes = len(np.unique(y))

    # Data Prep
    encoder = OneHotEncoder(sparse_output=False)
    y_onehot = encoder.fit_transform(y.reshape(-1, 1))

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X.astype(np.float32), y_onehot.astype(np.float32), test_size=0.2, stratify=y, random_state=seed
    )

    val_frac = max(0.1, (num_classes + 5) / len(X_train_full))

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=val_frac, stratify=np.argmax(y_train_full, axis=1), random_state=seed
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
    X_val_scaled = scaler.transform(X_val).astype(np.float32)
    X_test_scaled = scaler.transform(X_test).astype(np.float32)
    
    # ---------------------------------------------------------
    # 1. Train NNEngine Model
    # ---------------------------------------------------------
    model = BenchmarkNet(num_features, hidden_size, num_classes)
    optimizer = nn.Adam(learning_rate=np.float32(lr))
    loss_fn = nn.SoftmaxCrossEntropyLoss()
    regularizer = nn.L2Regularizer(l2=np.float32(0.0001)) if use_l2 else None

    train_loader = nn.DataLoader(X_train_scaled, y_train, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = nn.DataLoader(X_val_scaled, y_val, batch_size=batch_size, shuffle=False)
    
    trainer = nn.JITCompiler(model, optimizer, loss_fn, regularizer)

    t0 = time.perf_counter()
    trainer.fit(train_loader, epochs=epochs, val_dataloader=val_loader, tol=1e-4, n_iter_no_change=10, verbose=False)
    t1 = time.perf_counter()

    nn_pred_logits = model.predict(X_test_scaled)
    nn_preds = np.argmax(nn_pred_logits, axis=1)
    y_test_labels = np.argmax(y_test, axis=1)

    nn_time = t1 - t0
    nn_acc = accuracy_score(y_test_labels, nn_preds) * 100
    
    # ---------------------------------------------------------
    # 2. Test Native C++ Checkpointing (named_parameters)
    # ---------------------------------------------------------
    ckpt_path = f"checkpoint_{name.replace(' ', '_').lower()}.nne"
    model.save_weights(ckpt_path)
    
    # Create a completely fresh, untrained model
    restored_model = BenchmarkNet(num_features, hidden_size, num_classes)
    restored_model.load_weights(ckpt_path)
    
    # Predict using the loaded weights
    restored_pred_logits = restored_model.predict(X_test_scaled)
    
    # Verify exact match (Proves serialization and Dropout Eval state work flawlessly)
    assert np.allclose(nn_pred_logits, restored_pred_logits, atol=1e-6), f"❌ Serialization failed for {name}!"
    os.remove(ckpt_path) # Cleanup

    # ---------------------------------------------------------
    # 3. Train Scikit-Learn Model
    # ---------------------------------------------------------
    sk_model = MLPClassifier(
        hidden_layer_sizes=(hidden_size,),
        activation="relu",
        solver="adam",
        batch_size=batch_size,
        learning_rate_init=np.float32(lr),
        max_iter=epochs,
        early_stopping=True,       
        validation_fraction=val_frac,
        n_iter_no_change=10,
        tol=1e-4,
        alpha=np.float32(0.0001) if use_l2 else np.float32(0.0),
        random_state=seed,
    )

    t0 = time.perf_counter()
    sk_model.fit(scaler.transform(X_train_full).astype(np.float32), np.argmax(y_train_full, axis=1))
    sk_preds = sk_model.predict(X_test_scaled)
    t1 = time.perf_counter()

    sk_time = t1 - t0
    sk_acc = accuracy_score(y_test_labels, sk_preds) * 100
    
    # ---------------------------------------------------------
    # 4. Print Results
    # ---------------------------------------------------------
    print(f"🚀 NNEngine (JIT)   — Accuracy: {nn_acc:.2f}%  |  Time: {nn_time:.4f}s")
    print(f"🐢 Sklearn (Adam)   — Accuracy: {sk_acc:.2f}%  |  Time: {sk_time:.4f}s")
    print(f"⚡ Speedup          — {sk_time / nn_time:.2f}x Faster")
    print(f"💾 Checkpoint       — Verified & Passed ✓")

def main() -> None:
    parser = argparse.ArgumentParser(description="Run NNEngine JIT benchmark datasets.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    print("🔥 NNEngine Phase 2: PyTorch UX, Dropout, and Checkpointing\n")

    iris = load_iris()
    test_dataset("Iris Flower", iris.data, iris.target, epochs=100, lr=np.float32(0.01), batch_size=16, hidden_size=16, seed=args.seed)

    digits = load_digits()
    test_dataset("Handwritten Digits", digits.data, digits.target, epochs=100, lr=np.float32(0.001), batch_size=32, hidden_size=128, seed=args.seed + 1)

    faces = fetch_olivetti_faces()
    test_dataset("Olivetti Faces", faces.data, faces.target, epochs=100, lr=np.float32(0.001), batch_size=32, hidden_size=128, seed=args.seed + 2)

if __name__ == "__main__":
    main()