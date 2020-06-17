import dataloader

dl = dataloader.DataLoader(10)

for i, (src, tgt) in enumerate(dl):
    print(i, len(src), len(tgt), len(src[0] + tgt[0]))
