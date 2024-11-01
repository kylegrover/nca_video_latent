import torch
import torch.nn as nn
import torch.nn.functional as F

class NCABlock(nn.Module):
    def __init__(self, channels):
        super(NCABlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(128, channels, kernel_size=1)
        self.activation = nn.ReLU(inplace=True)
    
    def forward(self, x):
        out = self.activation(self.conv1(x))
        out = self.conv2(out)
        return out

class NCA(nn.Module):
    def __init__(self, input_channels=3, hidden_channels=16, num_steps=10):
        super(NCA, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.num_steps = num_steps
        
        self.conv = nn.Conv2d(input_channels + hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.nca_block = NCABlock(hidden_channels)
    
    def forward(self, x, steps=None):
        if steps is None:
            steps = self.num_steps
        for _ in range(steps):
            # Apply NCA block
            delta = self.nca_block(x)
            x = x + delta
            # Clamp to [0, 1]
            x = torch.clamp(x, 0.0, 1.0)
        return x
