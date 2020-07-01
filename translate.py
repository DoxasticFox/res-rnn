#!/usr/bin/env python3

import dataloader
import itertools
import nnmodules
import torch

# Check Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 16

src_enc = nnmodules.ResRnn(
    input_width=8, state_width=400, output_width=0, checkpoint_name='src_enc')
src_dec = nnmodules.ResRnn(
    input_width=8, state_width=400, output_width=8, checkpoint_name='src_dec')
tgt_enc = nnmodules.ResRnn(
    input_width=8, state_width=400, output_width=0, checkpoint_name='tgt_enc')
tgt_dec = nnmodules.ResRnn(
    input_width=8, state_width=400, output_width=8, checkpoint_name='tgt_dec')

src_enc = src_enc.to(device)
src_dec = src_dec.to(device)
tgt_enc = tgt_enc.to(device)
tgt_dec = tgt_dec.to(device)

optimizer = torch.optim.SGD(
    list(src_enc.parameters()) + \
    list(src_dec.parameters()) + \
    list(tgt_enc.parameters()) + \
    list(tgt_dec.parameters()),
    lr=1e+5,
    momentum=0.9
)

def model(b, training):
    empty_srcs = torch.zeros((b.srcs.size(0), b.srcs.size(1), 8)).to(device)
    empty_tgts = torch.zeros((b.tgts.size(0), b.tgts.size(1), 8)).to(device)

    # Create masks
    src_lens_tiled  = b.src_lens.view(1, b.srcs.size(1), 1).expand(b.srcs.size())
    tgt_lens_tiled  = b.tgt_lens.view(1, b.tgts.size(1), 1).expand(b.tgts.size())

    src_indices = torch.arange(b.srcs.size(0)) \
        .to(device).view(b.srcs.size(0), 1, 1).expand(b.srcs.size())
    tgt_indices = torch.arange(b.tgts.size(0)) \
        .to(device).view(b.tgts.size(0), 1, 1).expand(b.tgts.size())

    src_mask = (src_indices < src_lens_tiled).float()
    tgt_mask = (tgt_indices < tgt_lens_tiled).float()

    # Run forward pass
    _, state_s = src_enc(b.rsrcs, seq_indices=b.src_lens - 1)
    _, state_t = tgt_enc(b.rtgts, seq_indices=b.tgt_lens - 1)

    #if training:
    #    state_s = state_s + torch.randn_like(state_s) * 0.1
    #    state_t = state_t + torch.randn_like(state_t) * 0.1

    unmasked_output_s_s, _ = src_dec(empty_srcs, state=state_s, seq_indices=None)
    unmasked_output_t_t, _ = tgt_dec(empty_tgts, state=state_t, seq_indices=None)

    masked_output_s_s = unmasked_output_s_s * src_mask
    masked_output_t_t = unmasked_output_t_t * tgt_mask

    batch_loss = (
        torch.nn.functional.smooth_l1_loss(
            state_s,
            state_t,
            reduction='sum',
        ) / state_s.size(-2) * 1e-2 +
        torch.nn.functional.smooth_l1_loss(
            masked_output_s_s,
            b.srcs,
            reduction='sum',
        ) / src_mask.sum() +
        torch.nn.functional.smooth_l1_loss(
            masked_output_t_t,
            b.tgts,
            reduction='sum',
        ) / tgt_mask.sum()
    )

    return (
        empty_srcs,
        empty_tgts,
        unmasked_output_s_s,
        unmasked_output_t_t,
        state_s,
        state_t,
        batch_loss,
    )

# Train the model
batch_gen_args = dict(
    batch_size=batch_size,
    min_line_len=2,
    max_line_len=25,
    device=device,
)
batches = iter(dataloader.BatchGenerator(**batch_gen_args))

ema_weight = 0.9
ema_batch_loss = None

for i in itertools.count():
    b = next(batches)

    if i % 100 == 0:
        with torch.no_grad():
            (
                empty_srcs,
                empty_tgts,
                unmasked_output_s_s,
                unmasked_output_t_t,
                state_s,
                state_t,
                batch_loss,
            ) = model(b, training=False)
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

        unmasked_output_s_t, _ = tgt_dec(empty_tgts, state=state_s, seq_indices=None)
        unmasked_output_t_s, _ = src_dec(empty_srcs, state=state_t, seq_indices=None)
        print(
            'Input (srcs):',
            dataloader.tensor_2_string(b.srcs.permute(1, 0, 2)[0])
        )
        print(
            'Output (s_s):',
            dataloader.tensor_2_string(unmasked_output_s_s.permute(1, 0, 2)[0])
        )
        print(
            'Output (t_s):',
            dataloader.tensor_2_string(unmasked_output_t_s.permute(1, 0, 2)[0])
        )
        print(
            'Input (tgts):',
            dataloader.tensor_2_string(b.tgts.permute(1, 0, 2)[0])
        )
        print(
            'Output (t_t):',
            dataloader.tensor_2_string(unmasked_output_t_t.permute(1, 0, 2)[0])
        )
        print(
            'Output (s_t):',
            dataloader.tensor_2_string(unmasked_output_s_t.permute(1, 0, 2)[0])
        )
        print()

    (
        empty_srcs,
        empty_tgts,
        unmasked_output_s_s,
        unmasked_output_t_t,
        state_s,
        state_t,
        batch_loss,
    ) = model(b, training=True)

    # Backprpagation and optimization
    optimizer.zero_grad()
    batch_loss.backward()
    torch.nn.utils.clip_grad_norm_(
        list(src_enc.parameters()) + \
        list(src_dec.parameters()) + \
        list(tgt_enc.parameters()) + \
        list(tgt_dec.parameters()),
        1e-4
    )
    optimizer.step()

    if i % 10000 == 0 and i > 0:
        src_enc.save_checkpoint()
        src_dec.save_checkpoint()
        tgt_enc.save_checkpoint()
        tgt_dec.save_checkpoint()
