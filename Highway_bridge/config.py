from dataclasses import dataclass


@dataclass
class DataConfig:
    root: str
    batch_size: int = 16
    num_workers: int = 6
    superpoint_min_points: int = 20
    superpoint_eps: float = 0.1

@dataclass
class ModelConfig:
    in_channels: int = 32  # 输入特征维度
    hidden_channels: int = 256
    num_classes: int = 5
    num_layers: int = 4
    num_heads: int = 8
    dropout: float = 0.1

@dataclass
class TrainConfig:
    epochs: int = 200
    lr: float = 0.001
    weight_decay: float = 0.0001
    warmup_epochs: int = 10
    scheduler_type: str = 'cosine'
    save_dir: str = 'checkpoints'
    log_dir: str = 'logs'

@dataclass
class Config:
    data: DataConfig
    model: ModelConfig
    train: TrainConfig