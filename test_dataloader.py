import dataloader

dl = dataloader.DataLoader(200, max_line_len=1000)

for i, ((srcs, src_lens), (tgts, tgt_lens)) in enumerate(dl):
    print(i, srcs.size(), tgts.size())
