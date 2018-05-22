import random
import torch
import torch.nn as tnn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

# ---------------------------------------------------------------
# Handle dtypes for the device
USE_CUDA = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if USE_CUDA else torch.ByteTensor
Tensor = FloatTensor

# ---------------------------------------------------------------


class ReplayMemory(object):
    """A very generic memory system, with a finite capacity."""

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = args
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
