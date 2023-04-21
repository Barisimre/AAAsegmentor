import torch


def main():
    device = "cuda"


    example = torch.rand(size=(4, 256, 8, 8, 8)).to(device)
    conv = torch.nn.ConvTranspose3d(kernel_size=16, stride=16, in_channels=256, out_channels=8).to(device)

    ts = []
    for _ in range(500):
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        out = conv(example)
        end.record()

        torch.cuda.synchronize()
        t = start.elapsed_time(end)
        ts.append(t)

    print(sum(ts)/len(ts))


if __name__ == '__main__':
    main()
