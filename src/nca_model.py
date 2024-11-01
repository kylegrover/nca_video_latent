# src/nca_model.py
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
        
        self.nca_block = NCABlock(hidden_channels)
        self.decoder = nn.Conv2d(hidden_channels, input_channels, kernel_size=1)
    
    def forward(self, hidden_state, steps=None):
        """
        Args:
            hidden_state (torch.Tensor): Tensor of shape [B, hidden_channels, H, W]
            steps (int, optional): Number of update steps. Defaults to self.num_steps.
        
        Returns:
            torch.Tensor: Generated frame of shape [B, input_channels, H, W]
            torch.Tensor: Updated hidden state of shape [B, hidden_channels, H, W]
        """
        if steps is None:
            steps = self.num_steps
        for _ in range(steps):
            delta = self.nca_block(hidden_state)
            hidden_state = hidden_state + delta
            hidden_state = torch.clamp(hidden_state, 0.0, 1.0)
        output = self.decoder(hidden_state)
        return output, hidden_state
