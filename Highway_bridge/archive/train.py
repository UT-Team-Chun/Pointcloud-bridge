import logging
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.optim as optim
import wandb
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from config import *
from datasets.pcd import PointCloudDataset
from models.spt import SuperPointTransformer


def train(cfg: Config):
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # 创建保存目录
    save_dir = Path(cfg.train.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 初始化wandb
    wandb.init(project="superpoint-transformer", config=cfg)
    
    # 创建数据集和加载器
    train_dataset = PointCloudDataset(
        cfg.data.root,
        split='train'
    )
    val_dataset = PointCloudDataset(
        cfg.data.root,
        split='val'
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers
    )
    
    # 创建模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SuperPointTransformer(
        in_channels=cfg.model.in_channels,
        hidden_channels=cfg.model.hidden_channels,
        num_classes=cfg.model.num_classes,
        num_layers=cfg.model.num_layers,
        num_heads=cfg.model.num_heads,
        dropout=cfg.model.dropout
    ).to(device)
    
    # 优化器和学习率调度器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.train.epochs
    )
    
    # 训练循环
    best_val_acc = 0.0
    for epoch in range(cfg.train.epochs):
        # 训练
        model.train()
        train_loss = 0.0
        train_acc = 0.0

        total_samples = len(train_dataset)
        pqbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{cfg.train.epochs}',
                     total=total_samples//cfg.data.batch_size, position=0, leave=True)
        
        for batch in pqbar:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            out = model(batch)
            loss = F.cross_entropy(out, batch.y)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pred = out.argmax(dim=1)
            train_acc += (pred == batch.y).float().mean().item()

            pqbar.set_postfix({
                'loss': loss.item(),
                'acc': (pred == batch.y).float().mean().item()
            })
        
        train_loss /= len(train_loader)
        train_acc /= len(train_loader)
        
        # 验证
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch)
                loss = F.cross_entropy(out, batch.y)
                
                val_loss += loss.item()
                pred = out.argmax(dim=1)
                val_acc += (pred == batch.y).float().mean().item()
        
        val_loss /= len(val_loader)
        val_acc /= len(val_loader)
        
        # 更新学习率
        scheduler.step()
        
        # 记录日志
        wandb.log({
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'lr': scheduler.get_last_lr()[0]
        })
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                model.state_dict(),
                save_dir / 'best_model.pth'
            )
        
        logger.info(
            f'Epoch {epoch+1}/{cfg.train.epochs} - '
            f'Train Loss: {train_loss:.4f} - '
            f'Train Acc: {train_acc:.4f} - '
            f'Val Loss: {val_loss:.4f} - '
            f'Val Acc: {val_acc:.4f}'
        )

if __name__ == '__main__':
    # 加载配置
    cfg = Config(
        data=DataConfig(root='data/all'),#path/to/your/data
        model=ModelConfig(),
        train=TrainConfig()
    )
    
    train(cfg)