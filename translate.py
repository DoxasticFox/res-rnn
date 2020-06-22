#!/usr/bin/env python3

import dataloader
import nnmodules
import torch

# Check Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batches = dataloader.BatchGenerator(
    batch_size=50,
    min_line_len=5,
    max_line_len=200
)

src_enc = nnmodules.ResRnn(
    input_width=8, state_width=1500, output_width=0, checkpoint_name='src_enc')
src_dec = nnmodules.ResRnn(
    input_width=0, state_width=1500, output_width=8, checkpoint_name='src_dec')
tgt_enc = nnmodules.ResRnn(
    input_width=8, state_width=1500, output_width=0, checkpoint_name='tgt_enc')
tgt_dec = nnmodules.ResRnn(
    input_width=0, state_width=1500, output_width=8, checkpoint_name='tgt_dec')

src_enc = src_enc.to(device)
src_dec = src_dec.to(device)
tgt_enc = tgt_enc.to(device)
tgt_dec = tgt_dec.to(device)

smooth_l1_loss = torch.nn.functional.smooth_l1_loss

optimizer = torch.optim.SGD(
    list(src_enc.parameters()) + \
    list(src_dec.parameters()) + \
    list(tgt_enc.parameters()) + \
    list(tgt_dec.parameters()),
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

    # Create masks
    src_lens_tiled  = src_lens.view(1, srcs.size(1), 1).expand(srcs.size())
    tgt_lens_tiled  = tgt_lens.view(1, tgts.size(1), 1).expand(tgts.size())

    src_indices = torch.arange(srcs.size(0)) \
        .to(device).view(srcs.size(0), 1, 1).expand(srcs.size())
    tgt_indices = torch.arange(tgts.size(0)) \
        .to(device).view(tgts.size(0), 1, 1).expand(tgts.size())

    src_mask = (src_indices < src_lens_tiled).float()
    tgt_mask = (tgt_indices < tgt_lens_tiled).float()

    # Run forward pass
    _, state_s = src_enc(srcs, seq_indices=src_lens - 1)
    _, state_t = tgt_enc(tgts, seq_indices=tgt_lens - 1)

    state_s_n = state_s + torch.randn_like(state_s).to(device) * 0.25
    state_t_n = state_t + torch.randn_like(state_t).to(device) * 0.25

    output_s_s, _ = src_dec(empty_srcs, state=state_s_n, seq_indices=None)
    output_t_t, _ = tgt_dec(empty_tgts, state=state_t_n, seq_indices=None)

    masked_output_s_s = output_s_s * src_mask
    masked_output_t_t = output_t_t * tgt_mask

    total_loss = \
        smooth_l1_loss(state_s, state_t) + \
        smooth_l1_loss(masked_output_s_s, srcs) + \
        smooth_l1_loss(masked_output_t_t, tgts)

    # Backprpagation and optimization
    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(
        list(src_enc.parameters()) + \
        list(src_dec.parameters()) + \
        list(tgt_enc.parameters()) + \
        list(tgt_dec.parameters()),
        0.0001
    )
    optimizer.step()

    if i % 10000 == 0 and i > 0:
        src_enc.save_checkpoint()
        src_dec.save_checkpoint()
        tgt_enc.save_checkpoint()
        tgt_dec.save_checkpoint()

    if (i + 1) % 10 == 0:
        print('Step {}, Total loss: {:.4f}'.format(i + 1, total_loss.item()))

    if i % 100 == 0:
        output_s_t, _ = tgt_dec(empty_tgts, state=state_s, seq_indices=None)
        output_t_s, _ = src_dec(empty_srcs, state=state_t, seq_indices=None)
        with torch.no_grad():
            print(
                'Input (srcs):',
                dataloader.tensor_2_string(srcs.permute(1, 0, 2)[0])
            )
            print(
                'Input (tgts):',
                dataloader.tensor_2_string(tgts.permute(1, 0, 2)[0])
            )

            print(
                'Output (s_s):',
                dataloader.tensor_2_string(output_s_s.permute(1, 0, 2)[0])
            )

            print(
                'Output (s_t):',
                dataloader.tensor_2_string(output_s_t.permute(1, 0, 2)[0])
            )

            print(
                'Output (t_s):',
                dataloader.tensor_2_string(output_t_s.permute(1, 0, 2)[0])
            )

            print(
                'Output (t_t):',
                dataloader.tensor_2_string(output_t_t.permute(1, 0, 2)[0])
            )
