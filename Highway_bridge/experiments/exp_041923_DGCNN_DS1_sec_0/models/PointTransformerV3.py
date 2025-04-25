import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from functools import partial

# 定义PointTransformerV3基本模块
class GEGLU(nn.Module):
    """门控GELU激活函数
    
    GEGLU: Gate-enhanced Gaussian Error Linear Unit
    参考自PTv3原始实现
    """
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)
        self.dim_out = dim_out

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)

class FeedForward(nn.Module):
    """多层感知机前馈网络
    
    PTv3使用的是改进版MLP块，包含GEGLU激活
    """
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            GEGLU(dim, hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class PositionalEncoding(nn.Module):
    """可学习的位置编码
    
    将点的坐标信息编码为高维嵌入
    """
    def __init__(self, d_model, scale_factor=1.0):
        super().__init__()
        self.scale_factor = scale_factor
        self.d_model = d_model
        self.linear = nn.Linear(3, d_model)
        
    def forward(self, xyz):
        """
        输入:
            xyz: 点云坐标 [B, N, 3]
        输出:
            pos_encoding: 位置编码 [B, N, d_model]
        """
        # 对坐标进行缩放，提高数值稳定性
        xyz_scaled = xyz * self.scale_factor
        # 使用MLP将3D坐标映射到高维特征空间
        pos_encoding = self.linear(xyz_scaled)
        return pos_encoding

class PointAttention(nn.Module):
    """点云注意力模块
    
    PTv3中的注意力机制，支持高效的Flash Attention实现
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., use_flash=True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.use_flash = use_flash and hasattr(F, "scaled_dot_product_attention")

    def forward(self, x, pos_encoding=None):
        """
        输入:
            x: 点云特征 [B, N, C]
            pos_encoding: 位置编码 [B, N, C]
        输出:
            x: 注意力后的特征 [B, N, C]
        """
        B, N, C = x.shape
        
        # 添加位置编码
        if pos_encoding is not None:
            x = x + pos_encoding
        
        # 计算QKV
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # [B, num_heads, N, head_dim]
        
        # 使用Flash Attention进行高效计算
        if self.use_flash:
            # 使用PyTorch 2.0+ 的 Flash Attention
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.0
            )
        else:
            # 普通注意力计算
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v
        
        x = x.transpose(1, 2).reshape(B, N, C)  # [B, N, C]
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x

class PointTransformerBlock(nn.Module):
    """点云Transformer块
    
    包含自注意力和前馈网络，使用Layer Normalization
    """
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), use_flash=True):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = PointAttention(
            dim=dim, num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop, use_flash=use_flash)
        
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = FeedForward(dim=dim, hidden_dim=mlp_hidden_dim, dropout=drop)

    def forward(self, x, pos_encoding=None):
        """
        输入:
            x: 点云特征 [B, N, C]
            pos_encoding: 位置编码 [B, N, C]
        输出:
            x: 处理后的特征 [B, N, C]
        """
        # 残差连接 + 自注意力
        x = x + self.attn(self.norm1(x), pos_encoding)
        # 残差连接 + 前馈网络
        x = x + self.mlp(self.norm2(x))
        return x

class PatchEmbed(nn.Module):
    """点云特征嵌入层
    
    将输入的点云特征转换为指定维度的嵌入特征
    """
    def __init__(self, in_chans, embed_dim=96, norm_layer=None):
        super().__init__()
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.proj = nn.Linear(in_chans, embed_dim)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        """
        输入:
            x: 点云特征 [B, N, in_chans]
        输出:
            x: 嵌入特征 [B, N, embed_dim]
        """
        x = self.proj(x)
        x = self.norm(x)
        return x

class PointTransformerV3(nn.Module):
    """PointTransformerV3主网络
    
    实现了论文中的结构，适用于点云语义分割任务
    """
    def __init__(self, num_classes=5, d_in=6, embed_dim=384, depth=12, 
                 num_heads=6, mlp_ratio=4., qkv_bias=True, drop_rate=0.1, 
                 attn_drop_rate=0.1, use_flash=True):
        """
        参数:
            num_classes: 分类类别数
            d_in: 输入特征维度，预设为6 (xyz + rgb)
            embed_dim: Transformer的嵌入维度
            depth: Transformer块的数量
            num_heads: 多头注意力中的头数
            mlp_ratio: MLP隐藏层维度与嵌入维度的比率
            qkv_bias: 是否使用QKV的偏置
            drop_rate: Dropout率
            attn_drop_rate: 注意力Dropout率
            use_flash: 是否使用Flash Attention加速
        """
        super().__init__()
        
        self.d_in = d_in  # 记录输入维度便于forward中检查
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        
        # 特征嵌入层
        self.patch_embed = PatchEmbed(
            in_chans=d_in,
            embed_dim=embed_dim,
            norm_layer=norm_layer
        )
        
        # 位置编码 (只对xyz进行编码)
        self.pos_embed = PositionalEncoding(embed_dim)
        
        # Transformer块
        self.blocks = nn.ModuleList([
            PointTransformerBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate,
                norm_layer=norm_layer, use_flash=use_flash
            )
            for i in range(depth)
        ])
        
        # 分类头
        self.norm = norm_layer(embed_dim)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
        # 初始化权重
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        """初始化模型权重"""
        if isinstance(m, nn.Linear):
            # 使用截断正态分布初始化线性层权重
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    def _check_input_dims(self, xyz, features):
        """检查输入维度并处理一致性问题"""
        batch_size, num_points, _ = xyz.shape
        
        # 处理输入特征
        if features is None:
            # 如果没有额外特征，使用坐标作为特征
            if self.d_in == 3:
                input_features = xyz
            else:
                # 打印警告
                print(f"警告: 模型期望 {self.d_in} 维输入特征，但只提供了xyz(3维)。将进行零填充。")
                padding = torch.zeros(batch_size, num_points, self.d_in - 3, device=xyz.device)
                input_features = torch.cat([xyz, padding], dim=2)
        else:
            # 如果提供了features，拼接xyz和features
            combined_features = torch.cat([xyz, features], dim=2)
            feat_dim = combined_features.shape[2]
            
            if feat_dim == self.d_in:
                input_features = combined_features
            elif feat_dim > self.d_in:
                print(f"警告: 特征维度({feat_dim})超过模型期望的输入维度({self.d_in})。进行截断。")
                input_features = combined_features[:, :, :self.d_in]
            else:
                print(f"警告: 特征维度({feat_dim})小于模型期望的输入维度({self.d_in})。进行零填充。")
                padding = torch.zeros(batch_size, num_points, self.d_in - feat_dim, device=xyz.device)
                input_features = torch.cat([combined_features, padding], dim=2)
                
        return input_features

    def forward(self, xyz, features=None):
        """
        输入:
            xyz: 点云坐标 [B, N, 3]
            features: 点云特征 [B, N, C-3] (通常是rgb颜色)
        输出:
            x: 每个点的类别预测 [B, N, num_classes]
        """
        B, N, _ = xyz.shape
        
        # 处理输入特征，确保维度正确
        x = self._check_input_dims(xyz, features)
        
        # 特征嵌入
        x = self.patch_embed(x)
        
        # 位置编码 (只对xyz进行编码)
        pos_encoding = self.pos_embed(xyz)
        
        # 通过Transformer块
        for block in self.blocks:
            x = block(x, pos_encoding)
        
        # 归一化和分类头
        x = self.norm(x)  # [B, N, embed_dim]
        
        # 应用分类头
        x_reshaped = x.reshape(-1, x.shape[-1])  # [B*N, embed_dim]
        x_reshaped = self.head(x_reshaped)  # [B*N, num_classes]
        x = x_reshaped.reshape(B, N, -1)  # [B, N, num_classes]
        
        return x 