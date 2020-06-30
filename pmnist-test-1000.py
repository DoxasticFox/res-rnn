#!/usr/bin/env python3

import torch
import torchvision
import torchvision.transforms as transforms
import nnmodules

# Check Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# MNIST dataset
test_dataset = torchvision.datasets.MNIST(root='data',
                                          train=False,
                                          transform=transforms.ToTensor())

# Data loader
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=10000,
                                          shuffle=False,
                                          pin_memory=True)

model = nnmodules.ResRnn(
    input_width=1,
    state_width=1000,
    output_width=10,
    linearity=0.99999,
).to(device)

model.load('models/pmnist-1-1000-10-98.28.pt')

torch.manual_seed(0)
random_indices = torch.randperm(28 * 28)

# Train the model
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.reshape(images.size(0), 28 * 28)
        images = images[:, random_indices]
        images = images.reshape(images.size(0), 28 * 28, 1)
        images = images.permute(1, 0, 2)
        images = images.to(device)

        labels = labels.to(device)
        outputs, _ = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))
