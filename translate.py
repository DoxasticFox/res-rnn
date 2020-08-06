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

src_enc = nnmodules.ResRnn(
    input_width=256,
    stream_width=512,
    output_width=0,
    checkpoint_name='src_enc'
).to(device)

src_dec = nnmodules.ResRnn(
    input_width=0,
    stream_width=512,
    output_width=256,
    checkpoint_name='src_dec'
).to(device)

tgt_enc = nnmodules.ResRnn(
    input_width=256,
    stream_width=512,
    output_width=0,
    checkpoint_name='tgt_enc'
).to(device)

tgt_dec = nnmodules.ResRnn(
    input_width=0,
    stream_width=512,
    output_width=256,
    checkpoint_name='tgt_dec'
).to(device)

src_enc.load('/home/christian/pytorch/checkpoints/src_enc-25/49.pt')
src_dec.load('/home/christian/pytorch/checkpoints/src_dec-25/49.pt')
tgt_enc.load('/home/christian/pytorch/checkpoints/tgt_enc-25/49.pt')
tgt_dec.load('/home/christian/pytorch/checkpoints/tgt_dec-25/49.pt')

parameters = \
    list(src_enc.parameters()) + \
    list(src_dec.parameters()) + \
    list(tgt_enc.parameters()) + \
    list(tgt_dec.parameters())

optimizer = torch.optim.SGD(
    parameters,
    lr=1e+8,
    momentum=0.9,
)

def reversed_bytes(bytes_):
    return bytes(reversed(bytes_))

def model_wrapper(b, c, translate=False):
    empty_srcs = torch.zeros((b.srcs.size(0), b.srcs.size(1), 0)).to(device)
    empty_tgts = torch.zeros((b.tgts.size(0), b.tgts.size(1), 0)).to(device)

    # Create masks
    src_lens_tiled  = b.src_lens.view(
        1, b.srcs.size(1), 1).expand(b.srcs.size())
    tgt_lens_tiled  = b.tgt_lens.view(
        1, b.tgts.size(1), 1).expand(b.tgts.size())

    src_indices = torch.arange(b.srcs.size(0)) \
        .to(device).view(b.srcs.size(0), 1, 1).expand(b.srcs.size())
    tgt_indices = torch.arange(b.tgts.size(0)) \
        .to(device).view(b.tgts.size(0), 1, 1).expand(b.tgts.size())

    src_mask = (src_indices < src_lens_tiled).float()
    tgt_mask = (tgt_indices < tgt_lens_tiled).float()

    # Run forward pass
    _, stream_src = src_enc(b.srcs, seq_indices=b.src_lens - 1)
    _, stream_tgt = tgt_enc(b.tgts, seq_indices=b.tgt_lens - 1)

    unmasked_output_src_src, _ = src_dec(
        empty_srcs,
        stream=stream_src * torch.bernoulli(0.9 * torch.ones_like(stream_src)),
        seq_indices=None)
    unmasked_output_tgt_tgt, _ = tgt_dec(
        empty_tgts,
        stream=stream_tgt * torch.bernoulli(0.9 * torch.ones_like(stream_tgt)),
        seq_indices=None)

    unmasked_output_src_tgt, _ = tgt_dec(
        empty_tgts, stream=stream_src, seq_indices=None)
    unmasked_output_tgt_src, _ = src_dec(
        empty_srcs, stream=stream_tgt, seq_indices=None)

    masked_output_src_src = unmasked_output_src_src * src_mask
    masked_output_tgt_tgt = unmasked_output_tgt_tgt * tgt_mask
    masked_output_src_tgt = unmasked_output_src_tgt * tgt_mask
    masked_output_tgt_src = unmasked_output_tgt_src * src_mask

    batch_loss = (
        torch.nn.functional.smooth_l1_loss(
            masked_output_src_src,
            c.srcs,
        ) +
        torch.nn.functional.smooth_l1_loss(
            masked_output_tgt_tgt,
            c.tgts,
        ) +

        # torch.nn.functional.smooth_l1_loss(
        #     masked_output_src_tgt,
        #     c.tgts,
        # ) +
        # torch.nn.functional.smooth_l1_loss(
        #     masked_output_tgt_src,
        #     c.srcs,
        # )

        torch.nn.functional.smooth_l1_loss(
            stream_src,
            stream_tgt,
        ) * 1e-1
        # 1e-4 / (torch.nn.functional.smooth_l1_loss(
        #     stream_src[
        #         (
        #             torch.arange(stream_src.size(0)) + 1
        #         ).remainder(stream_src.size(0))
        #     ],
        #     stream_tgt,
        # ))
    )

    return (
        unmasked_output_src_src,
        unmasked_output_tgt_tgt,
        unmasked_output_src_tgt,
        unmasked_output_tgt_src,
        batch_loss,
    )

# Train the model
batch_gen_args = dict(
    batch_size=batch_size,
    min_line_len=2,
    max_line_len=50,
    device=device,
)
batches = iter(dataloader.BatchGenerator(**batch_gen_args))

ema_weight = 0.9
ema_batch_loss = None

for i in itertools.count():
    b = next(batches)
    b, c = b.map(reversed_bytes), b

    if i % 100 == 0:
        with torch.no_grad():
            (
                unmasked_output_src_src,
                unmasked_output_tgt_tgt,
                unmasked_output_src_tgt,
                unmasked_output_tgt_src,
                batch_loss,
            ) = model_wrapper(b, c, translate=True)
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
        print()
        print(
            'Input  (src)     :',
            dataloader.tensor_2_strings(b.srcs[:, 0:1, :])[0][::-1],
        )
        print(
            'Output (src->src):',
            dataloader.tensor_2_strings(unmasked_output_src_src[:, 0:1, :])[0],
        )
        print(
            'Output (tgt->src):',
            dataloader.tensor_2_strings(unmasked_output_tgt_src[:, 0:1, :])[0],
        )
        print()
        print(
            'Input  (tgts)    :',
            dataloader.tensor_2_strings(b.tgts[:, 0:1, :])[0][::-1],
        )
        print(
            'Output (tgt->tgt):',
            dataloader.tensor_2_strings(unmasked_output_tgt_tgt[:, 0:1, :])[0],
        )
        print(
            'Output (src->tgt):',
            dataloader.tensor_2_strings(unmasked_output_src_tgt[:, 0:1, :])[0],
        )
        print()
        print()
        print()

    (
        unmasked_output_src_src,
        unmasked_output_tgt_tgt,
        unmasked_output_src_tgt,
        unmasked_output_tgt_src,
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
        src_enc.save_checkpoint()
        src_dec.save_checkpoint()
        tgt_enc.save_checkpoint()
        tgt_dec.save_checkpoint()
