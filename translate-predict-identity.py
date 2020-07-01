#!/usr/bin/env python3

import dataloader
import itertools
import nnmodules
import torch

# Check Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 16

tgt_enc = nnmodules.ResRnn(
    input_width=8, state_width=500, output_width=8, checkpoint_name='tgt_enc')
tgt_dec = nnmodules.ResRnn(
    input_width=8, state_width=500, output_width=8, checkpoint_name='tgt_dec')

tgt_enc = tgt_enc.to(device)
tgt_dec = tgt_dec.to(device)

optimizer = torch.optim.SGD(
    list(tgt_enc.parameters()) + \
    list(tgt_dec.parameters()),
    lr=1e+5,
    momentum=0.9,
)

def model(b):
    shift_tgts = torch.cat(
        (
            b.tgts[1:, :, :],
            torch.zeros((1, b.tgts.size(1), b.tgts.size(2))).to(device),
        ),
        dim=0
    ).to(device)


    empty_tgts = torch.zeros((b.tgts.size(0), b.tgts.size(1), 8)).to(device)

    # Create masks
    tgt_lens_tiled  = b.tgt_lens.view(1, b.tgts.size(1), 1).expand(b.tgts.size())

    tgt_indices = torch.arange(b.tgts.size(0)) \
        .to(device).view(b.tgts.size(0), 1, 1).expand(b.tgts.size())

    tgt_mask = (tgt_indices < tgt_lens_tiled).float()

    # Run forward pass
    unmasked_output_t, state_t = tgt_enc(b.tgts, seq_indices=None)

    state_t = state_t[
        b.tgt_lens - 1,
        torch.arange(b.tgts.size(1))
    ]

    #unmasked_output_t_t, _ = tgt_dec(empty_tgts, state=state_t, seq_indices=None)
    unmasked_output_t_t, _ = None, None

    masked_output_t   = unmasked_output_t   * tgt_mask
    #masked_output_t_t = unmasked_output_t_t * tgt_mask
    masked_output_t_t = None

    batch_loss = (
        torch.nn.functional.smooth_l1_loss(
            masked_output_t,
            shift_tgts,
            reduction='sum',
        ) / tgt_mask.sum()
        # torch.nn.functional.smooth_l1_loss(
        #     masked_output_t_t,
        #     b.tgts,
        #     reduction='sum',
        # ) / tgt_mask.sum()
    )

    return (
        empty_tgts,
        unmasked_output_t,
        unmasked_output_t_t,
        state_t,
        batch_loss,
    )

# Train the model
batch_gen_args = dict(
    batch_size=batch_size,
    min_line_len=2,
    max_line_len=100,
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
                empty_tgts,
                unmasked_output_t,
                unmasked_output_t_t,
                state_t,
                batch_loss,
            ) = model(b)
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
            dataloader.tensor_2_string(b.tgts.permute(1, 0, 2)[0])
        )
        print(
            'Output (t)  : ',
            dataloader.tensor_2_string(unmasked_output_t.permute(1, 0, 2)[0])
        )
        # print(
        #     'Output (t_t):',
        #     dataloader.tensor_2_string(unmasked_output_t_t.permute(1, 0, 2)[0])
        # )
        print()

    (
        empty_tgts,
        unmasked_output_t,
        unmasked_output_t_t,
        state_t,
        batch_loss,
    ) = model(b)

    # Backprpagation and optimization
    optimizer.zero_grad()
    batch_loss.backward()
    torch.nn.utils.clip_grad_norm_(
        list(tgt_enc.parameters()) + \
        list(tgt_dec.parameters()),
        1e-4
    )
    optimizer.step()
