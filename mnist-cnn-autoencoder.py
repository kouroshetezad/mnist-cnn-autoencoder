#MNIST Classification
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
import matplotlib.pyplot as plt
import numpy as np


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)   
        self.conv2 = nn.Conv2d(32, 64, 3, 1)  

        
        self.fc1 = nn.Linear(64*12*12, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)  
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train_model(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    start_time = time.time()
    correct = 0
    total = 0
    running_loss = 0.0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(output, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

    end_time = time.time()
    acc = 100. * correct / total
    train_time = end_time - start_time
    return acc, train_time

def test_model(model, device, test_loader, criterion):
    model.eval()
    correct = 0
    total = 0
    test_loss = 0.0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            test_loss += loss.item()
            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    acc = 100. * correct / total
    return acc

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 5


for epoch in range(1, epochs + 1):
    train_acc, train_time = train_model(model, device, train_loader, optimizer, criterion, epoch)
    test_acc = test_model(model, device, test_loader, criterion)
    print(f"Epoch {epoch}: Train Acc = {train_acc:.2f}%, Test Acc = {test_acc:.2f}%, Time = {train_time:.2f}s")

#autoencoder

class ConvolutionalAutoencoder(nn.Module):
    def __init__(self):
        super(ConvolutionalAutoencoder, self).__init__()
        
        
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=1, padding=1),  
            nn.ReLU(True),
            nn.Conv2d(32, 64, 3, stride=1, padding=1),  
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2)  
        )
       
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 2, stride=2), 
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 1, 3, stride=1, padding=1), 
            nn.Tanh() 
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    

def train_autoencoder(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    running_loss = 0.0
    for data, _ in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
     
        loss = criterion(output, data) 
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * data.size(0)

    avg_loss = running_loss / len(train_loader.dataset)
    return avg_loss

def test_autoencoder(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0.0
  
    original_images = []
    reconstructed_images = []

    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            output = model(data)
            loss = criterion(output, data)
            test_loss += loss.item() * data.size(0)
            
            if i == 0: 
                original_images.append(data[:8].cpu())
                reconstructed_images.append(output[:8].cpu())

    avg_loss = test_loss / len(test_loader.dataset)
    return avg_loss, original_images[0], reconstructed_images[0]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
autoencoder = ConvolutionalAutoencoder().to(device)

criterion_ae = nn.MSELoss() 
optimizer_ae = optim.Adam(autoencoder.parameters(), lr=0.001)
epochs_ae = 5

for epoch in range(1, epochs_ae + 1):
    start_time = time.time()
    train_loss = train_autoencoder(autoencoder, device, train_loader, optimizer_ae, criterion_ae, epoch)
    end_time = time.time()
    
    test_loss, original_samples, reconstructed_samples = test_autoencoder(autoencoder, device, test_loader, criterion_ae)
    
    print(f"Epoch {epoch}: Train MSE Loss = {train_loss:.6f}, Test MSE Loss (Reconstruction Error) = {test_loss:.6f}, Time = {end_time - start_time:.2f}s")

def visualize_reconstruction(original, reconstructed, num_images=8):
    original = original.numpy().squeeze()
    reconstructed = reconstructed.numpy().squeeze()
    
    plt.figure(figsize=(10, 3))
    
    for i in range(num_images):
        ax = plt.subplot(2, num_images, i + 1)
        plt.imshow(original[i], cmap='gray') 
        ax.set_title("Original")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
      
        ax = plt.subplot(2, num_images, i + 1 + num_images)
        plt.imshow(reconstructed[i], cmap='gray')
        ax.set_title("Reconstructed")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        
    plt.tight_layout()
    plt.show()

visualize_reconstruction(original_samples, reconstructed_samples, num_images=8)