#!/usr/bin/env python3

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import nnmodules
import random
import dataloader

# Check Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batches = dataloader.BatchGenerator(
    batch_size=16,
    min_line_len=5,
    max_line_len=50
)

tgt_encode = nnmodules.ResRnn(input_width=8, state_width=500, output_width=0)
tgt_decode = nnmodules.ResRnn(input_width=0, state_width=500, output_width=8)

tgt_encode = tgt_encode.to(device)
tgt_decode = tgt_decode.to(device)

smooth_l1_loss = torch.nn.functional.smooth_l1_loss

optimizer = torch.optim.SGD(
    list(tgt_encode.parameters()) + \
    list(tgt_decode.parameters()),
    lr=100000.0,
    momentum=0.9
)

# Train the model
for i, batch in enumerate(batches):
    srcs            = batch.srcs.to(device)
    src_lens        = batch.src_lens.to(device)

    tgts            = batch.tgts.to(device)
    tgt_lens        = batch.tgt_lens.to(device)

    empty_srcs = torch.empty((srcs.size(0), srcs.size(1), 0)).to(device)
    empty_tgts = torch.empty((tgts.size(0), tgts.size(1), 0)).to(device)

    _,       state = tgt_encode(tgts, seq_indices=tgt_lens - 1)
    state          = state + torch.randn_like(state).to(device) * 0.25
    outputs, _     = tgt_decode(empty_tgts, state=state, seq_indices=None)

    total_loss = smooth_l1_loss(outputs, tgts)

    # Backprpagation and optimization
    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(
        list(tgt_encode.parameters()) + \
        list(tgt_decode.parameters()),
        0.0001
    )
    optimizer.step()

    if (i + 1) % 10 == 0:
        print('Step {}, Total loss: {:.4f}'.format(i + 1, total_loss.item()))

    if i % 100 == 0:
        with torch.no_grad():
            print(
                'Input :',
                dataloader.tensor_2_string(tgts.permute(1, 0, 2)[0])
            )
            print(
                'Output:',
                dataloader.tensor_2_string(outputs.permute(1, 0, 2)[0])
            )
