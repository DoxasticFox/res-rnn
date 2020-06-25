#!/usr/bin/env python3

import dataloader
import itertools
import nnmodules
import torch

# Check Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 16

tgt_enc = nnmodules.ResRnn(
    input_width=8, state_width=500, output_width=0, checkpoint_name='tgt_enc')
tgt_dec = nnmodules.ResRnn(
    input_width=0, state_width=500, output_width=8, checkpoint_name='tgt_dec')

tgt_enc = tgt_enc.to(device)
tgt_dec = tgt_dec.to(device)

optimizer = torch.optim.SGD(
    list(tgt_enc.parameters()) + \
    list(tgt_dec.parameters()),
    lr=100000.0,
    momentum=0.9
)

def model(b):
    empty_srcs = torch.empty((b.srcs.size(0), b.srcs.size(1), 0)).to(device)
    empty_tgts = torch.empty((b.tgts.size(0), b.tgts.size(1), 0)).to(device)

    # Create masks
    src_lens_tiled  = b.src_lens.view(1, b.srcs.size(1), 1).expand(b.srcs.size())
    tgt_lens_tiled  = b.tgt_lens.view(1, b.tgts.size(1), 1).expand(b.tgts.size())

    src_indices = torch.arange(b.srcs.size(0)) \
        .to(device).view(b.srcs.size(0), 1, 1).expand(b.srcs.size())
    tgt_indices = torch.arange(b.tgts.size(0)) \
        .to(device).view(b.tgts.size(0), 1, 1).expand(b.tgts.size())

    src_mask = (src_indices < src_lens_tiled).float()
    tgt_mask = (tgt_indices < tgt_lens_tiled).float()

    # Forward
    _, state_t = tgt_enc(b.tgts, seq_indices=b.tgt_lens - 1)

    unmasked_output_t_t, _ = tgt_dec(empty_tgts, state=state_t, seq_indices=None)

    masked_output_t_t = unmasked_output_t_t * tgt_mask

    batch_loss = torch.nn.functional.smooth_l1_loss(
        masked_output_t_t,
        b.tgts,
        reduction='sum'
    ) / masked_output_t_t.size(-2)

    return unmasked_output_t_t, batch_loss

# Train the model
batch_gen_args = dict(
    batch_size=batch_size,
    max_line_len=10,
    device=device,
)
batches = iter(dataloader.BatchGenerator(**batch_gen_args))

ema_weight = 0.9
seq_size_inc_threshold = 0.5
ema_batch_loss_init = 2.0 * seq_size_inc_threshold
ema_batch_loss = ema_batch_loss_init
seq_size_inc = 2

for i in itertools.count():
    b = next(batches)

    if i % 100 == 0:
        tgt_enc.eval()
        tgt_dec.eval()
        with torch.no_grad():
            unmasked_output_t_t, batch_loss = model(b)
            batch_loss_item = batch_loss.item()

        ema_batch_loss = \
            ema_weight * ema_batch_loss + \
            (1.0 - ema_weight) * batch_loss_item

        print(
            (
                'Step {}, '
                'Batch loss: {:.4f}, '
                'Batch loss EMA: {:.4f}, '
                'Max line len: {}'
            ).format(
                i,
                batch_loss_item,
                ema_batch_loss,
                batch_gen_args['max_line_len'],
            )
        )

        print(
            'Input :',
            dataloader.tensor_2_string(b.tgts.permute(1, 0, 2)[0])
        )
        print(
            'Output:',
            dataloader.tensor_2_string(unmasked_output_t_t.permute(1, 0, 2)[0])
        )
        print()

    if ema_batch_loss < seq_size_inc_threshold:
        batch_gen_args['max_line_len'] += seq_size_inc
        batches = iter(dataloader.BatchGenerator(**batch_gen_args))
        ema_batch_loss = ema_batch_loss_init

    tgt_enc.train()
    tgt_dec.train()
    unmasked_output_t_t, batch_loss = model(b)

    # Backprpagation and optimization
    optimizer.zero_grad()
    batch_loss.backward()
    torch.nn.utils.clip_grad_norm_(
        list(tgt_enc.parameters()) + \
        list(tgt_dec.parameters()),
        1e-4
    )
    optimizer.step()

    # if i % 10000 == 0 and i > 0:
    #     tgt_enc.save_checkpoint()
    #     tgt_dec.save_checkpoint()
