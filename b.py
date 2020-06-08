import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from nnmodules import *

# Check Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define Hyper-parameters
input_size = 784
state_size = 1000
output_size = 10
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

class ResRNN(nn.Module):
    def __init__(self, input_size, state_size, output_size):
        super(NeuralNet, self).__init__()

        hidden_size = input_size + state_size

        self.res = Res(hidden_size)
        self.fc1 = nn.Linear(hidden_size, output_size)

        torch.nn.init.xavier_uniform_(self.fc1.weight, gain=1.0)

    def forward(self, i):
        # i:       (seq_size, batch_size, input_size)
        # s:       (seq_size, batch_size, state_size)
        # x:       (seq_size, batch_size, input_size + state_size)
        # returns: (batch_size, output_size)

        x = torch.cat((i, s), dim=2)

        for _ in range(i.size(0)):
            x = self.res(x)

        x = self.fc1(x)

        return x

model = NeuralNet(input_size, state_size, output_size).to(device)
for p in model.parameters():
    p.register_hook(lambda grad: torch.clamp(grad, -0.1, 0.1))

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

optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

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
