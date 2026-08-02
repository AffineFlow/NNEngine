import os
import io
import numpy as np
from PIL import Image
import nnengine as nne

def fetch_and_prepare_fer_subset(img_size=64):
    print("Fetching full facial expression dataset...")
    try:
        from datasets import load_dataset
    except ImportError:
        import subprocess
        import sys
        print("Installing required 'datasets' package...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "datasets", "pillow"])
        from datasets import load_dataset

    # Download and cache the full dataset split
    dataset = load_dataset("abhilash88/fer2013-enhanced", split="train")
    
    # 0: Angry, 1: Disgust, 2: Fear, 3: Happy, 4: Sad, 5: Surprise, 6: Neutral
    target_mapping = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6}
    
    images = []
    labels = []
    
    print("Preprocessing all available images across all emotions...")
    for item in dataset:
        emotion_id = item.get('emotion', item.get('label'))
        if emotion_id in target_mapping:
            img_data = item['image']
            
            if isinstance(img_data, dict):
                if 'bytes' in img_data and img_data['bytes'] is not None:
                    img = Image.open(io.BytesIO(img_data['bytes']))
                elif 'path' in img_data and img_data['path'] is not None:
                    img = Image.open(img_data['path'])
                else:
                    continue
            elif isinstance(img_data, (list, np.ndarray)):
                arr = np.array(img_data, dtype=np.uint8)
                if arr.ndim == 1:
                    try:
                        side = int(np.sqrt(arr.size))
                        arr = arr.reshape((side, side))
                    except Exception:
                        pass
                img = Image.fromarray(arr)
            elif hasattr(img_data, 'convert'):
                img = img_data
            else:
                continue

            if img.mode != 'L':
                img = img.convert('L')
            
            img_resized = img.resize((img_size, img_size), Image.Resampling.BILINEAR)
            img_np = np.array(img_resized, dtype=np.float32) / 255.0
            images.append(img_np)
            labels.append(target_mapping[emotion_id])

    X = np.array(images, dtype=np.float32)
    X = np.expand_dims(X, axis=1)
    
    num_classes = 7
    y = np.eye(num_classes, dtype=np.float32)[labels]
    
    print(f"Successfully processed full dataset: {X.shape[0]} training samples loaded.")
    return X, y

def train():
    nne.set_seed(42)
    
    img_h, img_w, num_classes = 64, 64, 7
    X_train, y_train = fetch_and_prepare_fer_subset(img_size=img_h)

    # Architecture matches wasm_wrapper.cpp precisely for tensor/weight compatibility
    class NNEngineDeepCNN(nne.Module):
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

    model = NNEngineDeepCNN(img_h, img_w, num_classes)
    optimizer = nne.Adam(learning_rate=0.001)
    optimizer.set_parameters(model.parameters())
    loss_fn = nne.SoftmaxCrossEntropyLoss()

    trainer = nne.JITCompiler(model, optimizer, loss_fn)
    dataloader = nne.DataLoader(X_train, y_train, batch_size=32, shuffle=True, drop_last=True)
    
    print("Training emotion classifier on full dataset...")
    # Increased epochs to 25 for deeper convergence on the full dataset
    trainer.fit(dataloader, epochs=25, verbose=True)

    # Save weights directly to web/public for Vite and production build pick-up
    os.makedirs("web/public", exist_ok=True)
    weights_path = "web/public/emotion_weights.nne"
    model.save_weights(weights_path)
    print(f"Weights successfully saved to {weights_path}")

if __name__ == "__main__":
    train()