import torch
import torch.nn as nn
import numpy as  np
import time
import os
import psutil
import gpustat
from contextlib import contextmanager

def dice(mask, out, index):
    mask = mask == index
    out = out == index
    return 2 * (torch.sum(mask * out) / (torch.sum(mask) + torch.sum(out))).item()


def dice_scores(out, mask):
    return np.array([dice(mask, out, i) for i in range(3)])


def count_learnable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

@contextmanager
def resource_monitor():
    # CPU and RAM usage monitoring
    process = psutil.Process(os.getpid())
    cpu_start = process.cpu_percent(interval=None)
    ram_start = process.memory_info().rss

    # GPU usage monitoring
    try:
        gpu_start = gpustat.new_query().jsonify()["gpus"][0]["memory.used"]
    except Exception as e:
        gpu_start = "Unable to retrieve GPU info: {}".format(e)

    # Time monitoring
    start_time = time.time()

    yield

    # Time monitoring
    elapsed_time = time.time() - start_time

    # CPU and RAM usage monitoring
    cpu_end = process.cpu_percent(interval=None)
    ram_end = process.memory_info().rss

    # GPU usage monitoring
    try:
        gpu_end = gpustat.new_query().jsonify()["gpus"][0]["memory.used"]
    except Exception as e:
        gpu_end = "Unable to retrieve GPU info: {}".format(e)

    print(f'Elapsed time: {elapsed_time} seconds')
    print(f'CPU used: {cpu_end - cpu_start} percent')
    print(f'RAM used: {ram_end - ram_start} bytes')

    if isinstance(gpu_start, float) and isinstance(gpu_end, float):
        print(f'GPU memory used: {gpu_end - gpu_start} MiB')
    else:
        print(f'GPU usage: {gpu_end}')