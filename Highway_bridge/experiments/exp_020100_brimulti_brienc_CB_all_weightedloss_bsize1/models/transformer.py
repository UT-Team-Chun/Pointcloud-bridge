import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing


class MLP(nn.Module):
    def __init__(self, channels, dropout=0.0):
        """
        简单的多层感知机
        Args:
            channels: 各层的通道数列表
            dropout: dropout率
        """
        super().__init__()

        layers = []
        for i in range(len(channels) - 1):
            layers.append(nn.Linear(channels[i], channels[i + 1]))
            if i < len(channels) - 2:  # 除了最后一层，都加入ReLU和Dropout
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class MultiHeadAttention(MessagePassing):
    def __init__(
            self,
            in_channels: int,
            num_heads: int,
            dropout: float = 0.1
    ):
        super().__init__(aggr='add', node_dim=0)

        self.in_channels = in_channels
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads
        assert self.head_dim * num_heads == in_channels, "in_channels must be divisible by num_heads"

        # 投影层
        self.q_proj = nn.Linear(in_channels, in_channels)
        self.k_proj = nn.Linear(in_channels, in_channels)
        self.v_proj = nn.Linear(in_channels, in_channels)
        self.o_proj = nn.Linear(in_channels, in_channels)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr=None):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr=None):
        # 计算注意力分数
        q = self.q_proj(x_i).view(-1, self.num_heads, self.head_dim)
        k = self.k_proj(x_j).view(-1, self.num_heads, self.head_dim)
        v = self.v_proj(x_j).view(-1, self.num_heads, self.head_dim)

        # 计算注意力权重
        attn = (q * k).sum(dim=-1) / self.head_dim ** 0.5

        # 如果有边特征，加入计算
        if edge_attr is not None:
            edge_embedding = self.edge_proj(edge_attr)
            attn = attn + edge_embedding

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # 应用注意力权重
        out = v * attn.unsqueeze(-1)
        return out.view(-1, self.in_channels)

    def update(self, aggr_out):
        return self.o_proj(aggr_out)


class TransformerEncoder(nn.Module):
    def __init__(
            self,
            hidden_channels: int,
            num_heads: int,
            dropout: float = 0.1
    ):
        super().__init__()

        self.attention = MultiHeadAttention(
            hidden_channels,
            num_heads,
            dropout
        )

        # Feed-forward network
        self.ffn = MLP([
            hidden_channels,
            hidden_channels * 4,
            hidden_channels
        ], dropout=dropout)

        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_channels)
        self.norm2 = nn.LayerNorm(hidden_channels)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr=None):
        # 自注意力层
        residual = x
        x = self.norm1(x)
        x = self.attention(x, edge_index, edge_attr)
        x = self.dropout(x)
        x = x + residual

        # Feed-forward层
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = self.dropout(x)
        x = x + residual

        return x