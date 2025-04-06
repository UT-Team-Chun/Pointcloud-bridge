import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

class MLP(nn.Module):
    def __init__(
        self,
        channels: List[int],
        dropout: float = 0.1,
        batch_norm: bool = True
    ):
        super().__init__()
        
        layers = []
        for i in range(len(channels) - 1):
            layers.append(nn.Linear(channels[i], channels[i + 1]))
            
            if i < len(channels) - 2:  # 不在最后一层添加BN和激活
                if batch_norm:
                    layers.append(nn.BatchNorm1d(channels[i + 1]))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
                
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.layers(x)