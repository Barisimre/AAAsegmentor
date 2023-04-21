import torch
from src.model.my_model import MyModel
from src.constants import *


def main():
    device = "cuda"

    example = torch.rand(size=(2, 1, 128, 128, 128)).to(device)
    mask = torch.rand(size=(2, 3, 128, 128, 128)).to(device)

    model = MyModel(in_channels=1, out_channels=3, skip_transformer=False, channels=(16, 16, 16, 16, 16),
                    transformer_channels=8, embed_dim=256).to(DEVICE)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATES[0])

    ts = []
    for _ in range(5):
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        out = model(example)
        loss = LOSS(out, mask)
        loss.backward()
        optimizer.step()

        end.record()

        torch.cuda.synchronize()
        t = start.elapsed_time(end)
        ts.append(t)

    print(sum(ts) / len(ts))
    raise Exception(sum(ts) / len(ts))


if __name__ == '__main__':
    main()
