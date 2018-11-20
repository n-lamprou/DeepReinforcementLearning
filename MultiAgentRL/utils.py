import random
import torch
import numpy as np

def transpose_list(mylist):
    return list(map(list, zip(*mylist)))

def transpose_to_tensor(input_list):
    make_tensor = lambda x: torch.tensor(x, dtype=torch.float)
    return list(map(make_tensor, zip(*input_list)))

def soft_update(target, source, tau):
    """
    Perform DDPG soft update (move target params toward source based on weight factor tau)
    :param target (torch.nn.Module): Network to copy parameters to
    :param source (torch.nn.Module): Network whose parameters to copy
    :param tau (float, 0 < x < 1): Weight factor for update
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
