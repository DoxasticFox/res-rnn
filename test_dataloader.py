import dataloader

dl = dataloader.BatchGenerator(200, max_line_len=1000)

for i, batch in enumerate(dl):
    print(i, dataloader.tensor_2_string(batch.tgts.permute(1, 0, 2)[0]))
