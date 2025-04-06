import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from .transformer import TransformerEncoder
from .layers import MLP

class SuperPointTransformer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_classes: int,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.input_proj = MLP([in_channels, hidden_channels, hidden_channels])
        
        # Transformer编码器层
        self.transformer_layers = nn.ModuleList([
            TransformerEncoder(
                hidden_channels,
                num_heads,
                dropout=dropout
            ) for _ in range(num_layers)
        ])
        
        # 输出投影
        self.output_proj = MLP([
            hidden_channels,
            hidden_channels // 2,
            num_classes
        ])
        
    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        # 初始特征投影
        x = self.input_proj(x)
        
        # Transformer层
        for transformer in self.transformer_layers:
            x = transformer(x, edge_index, edge_attr)
        
        # 输出投影
        out = self.output_proj(x)
        
        return out