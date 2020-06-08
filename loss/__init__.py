import torch

class ClippedMse(torch.nn.Module):
    def __init__(self):
        super(ClippedMse, self).__init__()

    def forward(self, x, y):
        diff = x - y
        clipped_square = \
            torch.clamp(diff.pow(2), max=1/4) + \
            torch.clamp(- 1/4 + diff, min=1/4) + \
            torch.clamp(- 1/4 - diff, min=1/4) - \
            1/2
        return torch.mean(clipped_square)
