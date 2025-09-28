import torch
import random
def add_gaussian_noise(tensor, mean=0, std=1):
    noise = torch.randn(tensor.size()).to(DEVICE) * std + mean
    return tensor + noise

def add_uniform_noise(tensor, low=-0.1, high=0.1):
    noise = torch.rand(tensor.size()).to(DEVICE) * (high - low) + low
    return tensor + noise

def add_random_noise(tensor):
    noise_type = random.choice(['gaussian', 'uniform'])
    if noise_type == 'gaussian':
        return add_gaussian_noise(tensor, mean=0, std=0.1)
    elif noise_type == 'uniform':
        return add_uniform_noise(tensor, low=-0.1, high=0.1)