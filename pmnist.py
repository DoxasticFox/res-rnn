#!/usr/bin/env python3

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import nnmodules

# Check Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define Hyper-parameters
input_size = 784
state_size = 1000
output_size = 10
num_epochs = 10000
train_batch_size = 100
test_batch_size = 100

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
                                           batch_size=train_batch_size,
                                           shuffle=True,
                                           pin_memory=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=test_batch_size,
                                          shuffle=False,
                                          pin_memory=True)

model = nnmodules.ResRnn(
    input_width=1,
    stream_width=state_size,
    output_width=output_size,
).to(device)

# Loss and optimizer
def loss_fn(outputs, labels):
    one_hot = torch.nn.functional.one_hot(labels, num_classes=output_size).type(outputs.dtype)

    return torch.nn.functional.smooth_l1_loss(outputs, one_hot)

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=1e-4,
    momentum=0.9,
)

# Train the model
total_step = len(train_loader)
step_num = 0

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.reshape(images.size(0), 28 * 28, 1)
        images = images.permute(1, 0, 2)
        images = images.to(device)

        labels = labels.to(device)

        # Forward pass
        outputs, state = model(images)
        total_loss = loss_fn(outputs, labels)

        # Backprpagation and optimization
        optimizer.zero_grad()
        total_loss.backward()
        #torch.nn.utils.clip_grad_norm_(model.parameters(), 1e-1)
        optimizer.step()
        step_num += 1

        if step_num % 10 == 0:
            print(
                'Epoch [{}/{}], '
                'Step [{}/{}], '
                'Total loss: {:.4f}'
                .format(
                    epoch+1,
                    num_epochs,
                    i+1,
                    total_step,
                    total_loss.item()
                )
            )

        # Test the model
        # In the test phase, don't need to compute gradients (for memory efficiency)
        if step_num % 600 == 0:
            with torch.no_grad():
                correct = 0
                total = 0
                for images_, labels_ in test_loader:
                    images_ = images_.reshape(images_.size(0), 28 * 28, 1)
                    images_ = images_.permute(1, 0, 2)
                    images_ = images_.to(device)

                    labels_ = labels_.to(device)
                    outputs, _ = model(images_)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels_.size(0)
                    correct += (predicted == labels_).sum().item()

                print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))
