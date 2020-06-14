#!/usr/bin/env python3

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import nnmodules
import loss
import matplotlib.pyplot as plt

# Check Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define Hyper-parameters
input_size = 784
state_size = 256
output_size = 10
num_epochs = 10000
train_batch_size = 500
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
                                           pin_memory=True,
                                           num_workers=4)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=test_batch_size,
                                          shuffle=False,
                                          pin_memory=True,
                                          num_workers=4)

class EncoderDecoderRnn(torch.nn.Module):
    def __init__(self):
        super(EncoderDecoderRnn, self).__init__()

        self.encoder = nnmodules.ResRnn(
            input_width=1,
            state_width=state_size,
            output_width=1 + state_size
        )
        self.decoder = nnmodules.ResRnn(
            input_width=0,
            state_width=1 + state_size,
            output_width=28
        )

    def forward(self, input):
        encoded = self.encoder(input, output_indices=-1)
        decoded = self.decoder(encoded, max_iterations=28)
        return decoded

model = EncoderDecoderRnn().to(device)

smooth_l1_loss = torch.nn.SmoothL1Loss().to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=10000.0, momentum=0.9)

# Train the model
total_step = len(train_loader)
step_num = 0

for epoch in range(num_epochs):
    for i, (images, _) in enumerate(train_loader):
        images = images \
            .reshape(-1, 28 * 28, 1) \
            .expand(-1, -1, -1) \
            .permute(1, 0, 2) \
            .to(device)

        # Forward pass
        outputs = model(images)
        total_loss = smooth_l1_loss(
            images.permute(1, 0, 2).reshape(-1, 28, 28).permute(1, 0, 2),
            outputs
        )

        # Backprpagation and optimization
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.001)
        optimizer.step()

        if step_num % 10500 == 0:
            plt.imshow(outputs.permute(1, 0, 2)[0].cpu().detach().numpy(), cmap='gray_r')
            plt.show()

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
