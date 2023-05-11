import os
import torch
import gc
import psutil

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

def flush():
    gc.collect()
    torch.cuda.empty_cache()

def print_memory_usage():
    memory_info = psutil.virtual_memory()
    total_memory = memory_info.total / (1024 ** 2)  # Convert to MB
    used_memory = memory_info.used / (1024 ** 2)  # Convert to MB
    available_memory = memory_info.available / (1024 ** 2)  # Convert to MB
    percent_used = memory_info.percent

    print(f"Total memory: {total_memory:.2f} MB")
    print(f"Used memory: {used_memory:.2f} MB")
    print(f"Available memory: {available_memory:.2f} MB")
    print(f"Percent used: {percent_used}%")