import torch

class Orthonormality(torch.nn.Module):
    def __init__(self):
        super(Orthonormality, self).__init__()

    def forward(self, parameters):
        return torch.stack([self._single(p) for p in parameters]).sum()

    def _single(self, w):
        assert(len(set(w.size())) == 1)

        return (w.mm(w.t()) - self._eye_like(w)).pow(2.0).mean()

    def _eye_like(self, w):
        return torch.eye(*w.size()).to(w.device)
