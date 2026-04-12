import torch
import torch.nn as nn
import einops
from einops import reduce, rearrange
import math
import numpy as np

def data_loading(x, batch_size, context_length, device):
    
    random_starting_indices = torch.randint(0, len(x) - context_length, (batch_size,))

    # sequences of X, corresponding shifted by 1 sequences for y that are out targets
    x_seq = []
    for i in random_starting_indices:
        sample = torch.from_numpy((x[i : i + context_length]))
        x_seq.append(sample)
    
    y_seq = []
    for i in random_starting_indices:
        sample = torch.from_numpy((x[i + 1 : i + 1 + context_length]))
        y_seq.append(sample)
    
    X = torch.stack(x_seq)
    Y = torch.stack(y_seq)

    return X.to(device), Y.to(device)

def save_checkpoint(model, optimizer, iteration, out):
    checkpoint = {
        "model_state_dict" : model.state_dict(),
        "optimizer" : optimizer.state_dict(),
        "iteration" : iteration
    }
    torch.save(checkpoint, out)

def load_checkpoint(src, model, optimizer):
    checkpoint = torch.load(src, map_location="cpu")

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    return checkpoint["iteration"]