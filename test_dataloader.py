import torch
import dataloader

t = dataloader.strings_2_tensor([b'ab', b' !', b',.'])
s = dataloader.tensor_2_strings(t[:, 0:1, :])

print(t)
print(t.size())

print(s)
