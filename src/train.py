# src/train.py

import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import Flowers102
from torchvision.models import resnet18
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

def main():
    # Check GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Data transforms: resize + normalize
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # Load dataset (train + test split)
    train_data = Flowers102(root="data", split="train", download=True, transform=transform)
    test_data = Flowers102(root="data", split="test", download=True, transform=transform)

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=32)

    # Load pretrained ResNet
    model = resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 102)  # 102 flower classes
    model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop (just 1 epoch to start)
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    # Save model to file
    torch.save(model.state_dict(), "outputs/flower_resnet18.pth")
    print("💾 Model saved to outputs/flower_resnet18.pth")

if __name__ == "__main__":
    main()
