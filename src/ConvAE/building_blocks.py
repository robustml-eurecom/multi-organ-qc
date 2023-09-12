import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn

import torch
import torch.nn as nn

class ConvolutionalBlock(nn.Module):
    """
    Create a convolutional building block for a Convolutional Autoencoder in PyTorch.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels (filters) for the convolutional layers.
        kernel_size: Size of the convolutional kernels (e.g., 3 for 3x3 kernels).
        activation: Activation function to use after convolution (e.g., 'relu', 'sigmoid').
        pooling: Whether to add a MaxPooling2D layer after convolution.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, transpose=False, activation='relu', stride=1, pooling=True, is_dropout=True):
        super(ConvolutionalBlock, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=1) if not transpose else nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=1)  # Padding=1 for 'same' padding
        self.norm = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(.5)
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(.2)
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            raise ValueError("Unsupported activation function")

        self.pooling = pooling
        if pooling:
            self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
            
        self.is_dropout = is_dropout
        if is_dropout:
            self.dropout = nn.Dropout(.5)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        if self.pooling:
            x = self.maxpool(x)
        if self.is_dropout:
            x = self.dropout(x)
        return x
