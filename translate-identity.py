#!/usr/bin/env python3

import dataloader
import itertools
import math
import nnmodules
import random
import torch

# Check Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 16

enc = nnmodules.ResRnn(
    input_width=256,
    stream_width=8192,
    output_width=0,
    checkpoint_name='enc'
).to(device)

dec = nnmodules.ResRnn(
    input_width=0,
    stream_width=8192,
    output_width=256,
    checkpoint_name='dec'
).to(device)

#enc.load('/home/christian/pytorch/checkpoints/enc-6/4.pt')
#dec.load('/home/christian/pytorch/checkpoints/dec-6/4.pt')

parameters = list(enc.parameters()) + list(dec.parameters())

optimizer = torch.optim.SGD(
    parameters,
    lr=1e+8,
    momentum=0.9,
)

def underscores_like(bytes_):
    return b'_' * len(bytes_)

def random_word_delete(bytes_):
    return b' '.join(
        w if random.random() < 0.9 else underscores_like(w)
        for w in bytes_.split(b' ')
    )

def random_byte_delete(bytes_):
    return bytearray(
        b if random.random() < 0.8 else b'_'[0]
        for b in bytes_
    )

def model_wrapper(b, c):
    empty_tgts = torch.zeros((b.tgts.size(0), b.tgts.size(1), 0)).to(device)

    # Create masks
    tgt_lens_tiled  = b.tgt_lens.view(1, b.tgts.size(1), 1).expand(b.tgts.size())

    tgt_indices = torch.arange(b.tgts.size(0)) \
        .to(device).view(b.tgts.size(0), 1, 1).expand(b.tgts.size())

    tgt_mask = (tgt_indices < tgt_lens_tiled).float()

    # Run forward pass
    _, stream_t = enc(c.tgts, seq_indices=b.tgt_lens - 1)
    unmasked_output_t_t, _ = dec(empty_tgts, stream=stream_t, seq_indices=None)

    masked_output_t_t = unmasked_output_t_t * tgt_mask

    batch_loss = torch.nn.functional.smooth_l1_loss(
        masked_output_t_t,
        b.tgts,
    )

    return (
        unmasked_output_t_t,
        stream_t,
        batch_loss,
    )

# Train the model
batch_gen_args = dict(
    batch_size=batch_size,
    similar_lengths=True,
    min_line_len=2,
    max_line_len=50,
    device=device,
)
batches = iter(dataloader.BatchGenerator(**batch_gen_args))

ema_weight = 0.9
ema_batch_loss = None

for i in itertools.count():
    b = next(batches)
    c = b.map(random_byte_delete)

    if i % 10 == 0:
        with torch.no_grad():
            (
                unmasked_output_t_t,
                stream_t,
                batch_loss,
            ) = model_wrapper(b, c)
            batch_loss_item = batch_loss.item()

        ema_batch_loss = (
            batch_loss_item
            if ema_batch_loss is None
            else
            ema_weight * ema_batch_loss + (1.0 - ema_weight) * batch_loss_item
        )

        print(
            (
                'Step {}, '
                'Batch loss: {:.4f}, '
                'Batch loss EMA: {:.4f}'
            ).format(
                i,
                batch_loss_item,
                ema_batch_loss,
            )
        )

        print(
            'Input (tgts):',
            dataloader.tensor_2_strings(b.tgts[:, 0:1, :])[0],
        )
        print(
            'Input (tgts):',
            dataloader.tensor_2_strings(c.tgts[:, 0:1, :])[0],
        )
        print(
            'Output (t_t):',
            dataloader.tensor_2_strings(unmasked_output_t_t[:, 0:1, :])[0],
        )
        print()

    (
        unmasked_output_t_t,
        stream_t,
        batch_loss,
    ) = model_wrapper(b, c)

    # Backprpagation and optimization
    optimizer.zero_grad()
    batch_loss.backward()
    torch.nn.utils.clip_grad_norm_(
        parameters,
        1e-7,
    )
    optimizer.step()

    if i % 5000 == 0:
        enc.save_checkpoint()
        dec.save_checkpoint()
