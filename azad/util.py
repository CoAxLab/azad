import random
import torch
import torch.nn as tnn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T


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
