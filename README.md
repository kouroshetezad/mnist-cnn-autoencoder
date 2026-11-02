# MNIST CNN + Convolutional Autoencoder

PyTorch implementation of two models trained on the MNIST dataset:

1. **SimpleCNN** — a small convolutional classifier for handwritten digit recognition.
2. **ConvolutionalAutoencoder** — an encoder/decoder pair for image reconstruction and feature learning.

## Model details

### CNN classifier
- `conv1`: Conv2d(1 → 32, kernel 3)
- `conv2`: Conv2d(32 → 64, kernel 3)
- MaxPool2d(2) → flatten
- `fc1`: Linear(64·12·12 → 128)
- `fc2`: Linear(128 → 10)

Trained for 5 epochs with Adam (lr = 0.001), CrossEntropyLoss, batch size 64.

### Convolutional autoencoder
- Encoder: Conv2d(1 → 32) → ReLU → Conv2d(32 → 64) → ReLU → MaxPool2d(2)
- Decoder: ConvTranspose2d(64 → 32) → ReLU → ConvTranspose2d(32 → 1) → Tanh

Trained for 5 epochs with Adam (lr = 0.001) and MSE loss. Reconstructions of 8 test images are visualized with matplotlib.

## Requirements

```
torch
torchvision
matplotlib
numpy
```

## Run

```bash
python mnist-cnn-autoencoder.py
```

The MNIST dataset is downloaded automatically into `./data`. A CUDA GPU is used automatically when available.
