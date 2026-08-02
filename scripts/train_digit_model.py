import os
import numpy as np
import nnengine as nne

def fetch_mnist():
    print("Fetching MNIST dataset...")
    try:
        from datasets import load_dataset
    except ImportError:
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "datasets"])
        from datasets import load_dataset

    dataset = load_dataset("ylecun/mnist", split="train")
    
    images = []
    labels = []
    
    for item in dataset:
        # Convert to numpy array, normalize to [0.0, 1.0]
        img_np = np.array(item['image'], dtype=np.float32) / 255.0
        images.append(img_np)
        labels.append(item['label'])

    X = np.array(images, dtype=np.float32)
    X = np.expand_dims(X, axis=1) # Shape: (N, 1, 28, 28)
    
    y = np.eye(10, dtype=np.float32)[labels]
    
    print(f"Loaded {X.shape[0]} training samples.")
    return X, y

def train():
    nne.set_seed(42)
    img_h, img_w, num_classes = 28, 28, 10
    X_train, y_train = fetch_mnist()

    class NNEngineDigitCNN(nne.Module):
        def __init__(self, in_h, in_w, num_classes):
            super().__init__()
            # Same architecture, tailored for 28x28
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

    model = NNEngineDigitCNN(img_h, img_w, num_classes)
    optimizer = nne.Adam(learning_rate=0.001)
    optimizer.set_parameters(model.parameters())
    loss_fn = nne.SoftmaxCrossEntropyLoss()

    trainer = nne.JITCompiler(model, optimizer, loss_fn)
    dataloader = nne.DataLoader(X_train, y_train, batch_size=32, shuffle=True, drop_last=True)
    
    print("Training digit classifier...")
    trainer.fit(dataloader, epochs=5, verbose=True)

    os.makedirs("web/public", exist_ok=True)
    weights_path = "web/public/digit_weights.nne"
    model.save_weights(weights_path)
    print(f"Weights successfully saved to {weights_path}")

if __name__ == "__main__":
    train()