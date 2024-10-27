# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from pathlib import Path
from tqdm import tqdm
import logging
import datetime
import os


from models.enhanced_pointnet2 import EnhancedPointNet2
from utils.data_utils import BridgePointCloudDataset

class AverageMeter(object):
    """计算并存储平均值和当前值"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def setup_logging(log_dir):
    """设置日志"""
    log_dir = Path(log_dir)
    log_dir.mkdir(exist_ok=True)
    
    # 设置日志格式
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'training.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)



# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

# 配置参数
config = {
    'num_points': 4096,
    'batch_size': 32,
    'num_workers': 4,
    'learning_rate': 0.001,
    'num_epochs': 100,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

def train():
    # 创建实验目录
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_dir = Path(f'experiments/exp_{timestamp}')
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置tensorboard和日志
    writer = SummaryWriter(exp_dir / 'tensorboard')
    logger = setup_logging(exp_dir)
        
    # 记录配置
    logger.info("Training configuration:")
    for k, v in config.items():
        logger.info(f"{k}: {v}")
    
    # 设置设备
    device = torch.device(config['device'])
    logger.info(f'Using device: {device}')

    # 创建数据加载器
    train_dataset = BridgePointCloudDataset(
        data_dir='data/train',
        num_points=config['num_points'],
        transform=True
    )
    val_dataset = BridgePointCloudDataset(
        data_dir='data/val',
        num_points=config['num_points'],
        transform=False
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        num_workers=config['num_workers'],
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False, 
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Val dataset size: {len(val_dataset)}")
    
    # 创建模型
    model = EnhancedPointNet2(num_classes=8).to(device)
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['num_epochs'])
    
    # 训练循环
    best_val_loss = float('inf')
    best_val_acc = 0.0
    
    for epoch in range(config['num_epochs']):
        # 训练阶段
        model.train()
        train_loss = AverageMeter()
        train_acc = AverageMeter()
        
        # 创建进度条
        pbar = tqdm(train_loader, desc=f'Epoch [{epoch+1}/{config["num_epochs"]}] Training')
        
        for batch in pbar:
            points = batch['points'].to(device)
            colors = batch['colors'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            outputs = model(points, colors)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            # 计算准确率
            pred = outputs.max(1)[1]
            acc = pred.eq(labels).float().mean()
            
            # 更新统计
            train_loss.update(loss.item())
            train_acc.update(acc.item())
            
            # 更新进度条
            pbar.set_postfix({
                'Loss': f'{train_loss.avg:.4f}',
                'Acc': f'{train_acc.avg*100:.2f}%'
            })
        
        # 验证阶段
        model.eval()
        val_loss = AverageMeter()
        val_acc = AverageMeter()
        class_correct = torch.zeros(8).to(device)
        class_total = torch.zeros(8).to(device)
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validating'):
                points = batch['points'].to(device)
                colors = batch['colors'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(points, colors)
                loss = criterion(outputs, labels)
                
                # 计算准确率
                pred = outputs.max(1)[1]
                acc = pred.eq(labels).float().mean()
                
                # 更新统计
                val_loss.update(loss.item())
                val_acc.update(acc.item())
                
                # 计算每个类别的准确率
                for i in range(8):
                    mask = labels == i
                    class_correct[i] += pred[mask].eq(labels[mask]).sum()
                    class_total[i] += mask.sum()
        
        # 计算每个类别的准确率
        class_acc = class_correct / (class_total + 1e-6)
        
        # 记录训练信息
        logger.info(f"\nEpoch {epoch+1}/{config['num_epochs']}:")
        logger.info(f"Train Loss: {train_loss.avg:.4f}, Train Acc: {train_acc.avg*100:.2f}%")
        logger.info(f"Val Loss: {val_loss.avg:.4f}, Val Acc: {val_acc.avg*100:.2f}%")
        
        # 记录到tensorboard
        writer.add_scalar('Loss/train', train_loss.avg, epoch)
        writer.add_scalar('Loss/val', val_loss.avg, epoch)
        writer.add_scalar('Accuracy/train', train_acc.avg, epoch)
        writer.add_scalar('Accuracy/val', val_acc.avg, epoch)
        
        # 记录每个类别的准确率
        for i, acc in enumerate(class_acc):
            writer.add_scalar(f'Class_Accuracy/class_{i}', acc.item(), epoch)
            logger.info(f"Class {i} Accuracy: {acc.item()*100:.2f}%")
        
        # 保存最佳模型
        if val_acc.avg > best_val_acc:
            best_val_acc = val_acc.avg
            best_model_path = exp_dir / 'best_model.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': best_val_acc,
                'val_loss': val_loss.avg,
            }, best_model_path)
            logger.info(f"Saved best model with accuracy: {best_val_acc*100:.2f}%")
        
        # 保存最近的检查点
        checkpoint_path = exp_dir / 'latest_checkpoint.pth'
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_acc': val_acc.avg,
            'val_loss': val_loss.avg,
        }, checkpoint_path)
        
        scheduler.step()
    
    writer.close()
    logger.info("Training completed!")

if __name__ == '__main__':
    try:
        train()
    except Exception as e:
        logging.exception("Training failed with exception:")
        raise
