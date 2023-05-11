import os
import torch


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def unnormalize(tens, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    # assume tensor of shape NxCxHxW
    return tens * torch.Tensor(std)[None, :, None, None] + torch.Tensor(
        mean)[None, :, None, None]

def prune_parallel_trained_model(state_dict):
    new_state_dict = {}
    for key, value in state_dict['model'].items():
        new_key = key.replace('module.', '')  # Remove 'module.' from the key
        new_state_dict[new_key] = value
    return new_state_dict