#!/usr/bin/env python3

import torch
import torchvision
import torchvision.transforms as transforms
from nnmodules import *

# Check Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define Hyper-parameters
input_size = 784
hidden_size = 1000
output_width = 10
num_epochs = 10000
batch_size = 100

# MNIST dataset
train_dataset = torchvision.datasets.MNIST(root='data',
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='data',
                                          train=False,
                                          transform=transforms.ToTensor())

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           pin_memory=True,
                                           num_workers=4)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False,
                                          pin_memory=True,
                                          num_workers=4)

model = ShiftedResNet(input_size, hidden_size, output_width).to(device)
for p in model.parameters():
    p.register_hook(lambda grad: torch.clamp(grad, -0.01, 0.01))

def square(x):
    return \
            torch.clamp(x ** 2, - 1/2, 1/2) + \
            torch.max(torch.ones_like(x).to(device) / 2, - x) + \
            torch.max(torch.ones_like(x).to(device) / 2, x) - \
            1

# Loss and optimizer
def loss_fn(outputs, labels):
    one_hot = torch.zeros(labels.size(0), 10).to(device)
    one_hot[torch.arange(outputs.size(0)), labels] = 1

    output = torch.mean(square(outputs - one_hot))

    return output

optimizer = torch.optim.SGD(model.parameters(), lr=0.5, momentum=0.9)

# Train the model
total_step = len(train_loader)

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = loss_fn(outputs, labels)

        # Backprpagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

        # Test the model
        # In the test phase, don't need to compute gradients (for memory efficiency)
        if i % 600 == 0:
            with torch.no_grad():
                correct = 0
                total = 0
                for images_, labels_ in test_loader:
                    images_ = images_.reshape(-1, 28*28).to(device)
                    labels_ = labels_.to(device)
                    outputs = model(images_)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels_.size(0)
                    correct += (predicted == labels_).sum().item()

                print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))

# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')
