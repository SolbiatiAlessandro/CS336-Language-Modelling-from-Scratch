import torch
import numpy as np
from random import random

def data_loading_slow(x, batch_size, context_length, device='cpu'):
    xs, ys = [], []
    while len(xs) < batch_size: 
        prev = int(random() * (x.shape[0] - context_length))
        i = prev + context_length
        xs.append(torch.tensor(x[prev:i], device=device))
        ys.append(torch.tensor(x[prev+1:i+1], device=device))
    return torch.stack(xs), torch.stack(ys)


def data_loading(dataset, batch_size, context_length, device='cpu', dtype=torch.int):
    x_starts = np.random.randint(0, dataset.shape[0]-context_length, batch_size).reshape(-1, 1)
    offsets = np.arange(context_length).reshape(1, -1)
    x_index = offsets + x_starts
    y_index = x_index + 1
    return torch.tensor(dataset[x_index], device=device, dtype=dtype), torch.tensor(dataset[y_index], device=device, dtype=dtype)
