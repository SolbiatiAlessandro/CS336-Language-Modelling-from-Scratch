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


def data_loading_with_masking(dataset, offsets, prompt_lens, batch_size, context_length, device='cpu',
                              index_lo=0, index_hi=None):
    """
    dataset are tokenizer ids
    offsets are where the sample starts and ends
    prompt_len is where the actual answer starts
    Samples example indices from [index_lo, index_hi) (used to split train/val).
    index_hi defaults to num_examples-1 (excludes the final example, whose shifted
    target would overrun the flat array).
    Returns torch tensors on `device`: x (long), y (long), loss_mask (bool).
    """
    num_examples = offsets.shape[0] - 1
    if index_hi is None:
        index_hi = num_examples - 1
    starts = np.random.randint(index_lo, index_hi, batch_size)
    x = np.zeros((batch_size, context_length))
    loss_mask = np.zeros((batch_size, context_length))
    y = np.zeros((batch_size, context_length))
    for i, start in enumerate(starts):
        input_start = offsets[start]
        input_end = offsets[start+1]
        if input_end - input_start > context_length:
            continue
        _x = dataset[input_start:input_end]
        _y = dataset[input_start+1:input_end+1]
        x[i, 0:_x.shape[0]] = _x
        y[i, 0:_x.shape[0]] = _y
        loss_mask[i, int(prompt_lens[start])-1:_x.shape[0]-1] = 1
    x = torch.tensor(x, dtype=torch.long, device=device)
    y = torch.tensor(y, dtype=torch.long, device=device)
    loss_mask = torch.tensor(loss_mask, dtype=torch.bool, device=device)
    return x, y, loss_mask
