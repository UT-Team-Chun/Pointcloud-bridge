# train.py
import datetime
import logging
# tensorboard --logdir ./logs
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models.model import PointNet2
# from utils.data_utils import BridgePointCloudDataset
# from utils.BridgePCDataset import BridgePointCloudDataset
from utils.data_utils_ver2 import BridgePointCloudDataset
from utils.logger_config import initialize_logger

# 配置参数
config = {
    'num_points': 4096,
    'chunk_size': 4096,
    'overlap': 1024,
    'batch_size': 64,
    'num_workers': 0,
    'learning_rate': 0.001,
    'num_classes': 5,
    'num_epochs': 500,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}


def train():
    # 创建实验目录
    timestamp = datetime.datetime.now().strftime('%m%d_%H%M')
    case = 'pointnet2-iconpc-sepa'
    exp_dir = Path(f'experiments/exp_{case}_{timestamp}')
    exp_dir.mkdir(parents=True, exist_ok=True)

    # 设置tensorboard和日志
    writer = SummaryWriter(exp_dir / 'tensorboard')
    logger = initialize_logger(exp_dir)

    # 记录配置
    logger.info("Training configuration:")
    for k, v in config.items():
        logger.info(f"{k}: {v}")
    
    # 设置设备
    device = torch.device(config['device'])
    logger.info(f'Using device: {device}')

    # 创建数据加载器
    train_dataset = BridgePointCloudDataset(
        data_dir='data/fukushima/train/',
        num_points=config['num_points'],
        chunk_size = 4096,
        overlap=1024,
        #h_block_size=0.5,
        #v_block_size=0.5,
        #h_stride=0.4,
        #v_stride=0.4,
        #min_points=100,
        #logger=get_logger(),
        transform=True
    )
    logger.info('reading train data')


    val_dataset = BridgePointCloudDataset(
        data_dir='data/fukushima/val/',
        num_points=config['num_points'],
        chunk_size=4096,
        overlap = 1024,
        #h_block_size=0.5,
        #v_block_size=0.5,
        #h_stride=0.5,
        #v_stride=0.5,
        #min_points=100,
        #logger=get_logger()
    )

    logger.info('reading val data')

    # DataLoader的使用方式完全不变
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
    num_classes = config['num_classes']
    #model = EnhancedPointNet2(num_classes).to(device)
    model = PointNet2(num_classes).to(device)

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'],betas=(0.9, 0.999),
                      weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['num_epochs'], eta_min=1e-6)
    
    # 训练循环
    best_val_loss = float('inf')
    best_val_acc = 0.0

    # 计算类别权重
    #class_weights = compute_class_weights(train_loader,num_classes)
    #criterion = WeightedCrossEntropyLoss(weight=class_weights.to(device))

    
    for epoch in range(config['num_epochs']):
        # 训练阶段
        model.train()
        train_loss = AverageMeter()
        train_acc = AverageMeter()
        
        # 创建进度条
        total_samples = len(train_dataset)
        logger.info(f'Total samples is: {train_loader}')
        pbar = tqdm(train_loader, desc=f'Epoch [{epoch+1}/{config["num_epochs"]}] Training',
                    total=total_samples//config['batch_size'], position=0, leave=True)

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


        # 更新学习率
        scheduler.step()
        # 记录学习率
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('Learning_rate', current_lr, epoch)

        # Validation phase
        model.eval()
        val_loss = AverageMeter()
        val_acc = AverageMeter()
        confusion_matrix = torch.zeros(num_classes, num_classes).to(device) # 混淆矩阵
        class_correct = torch.zeros(num_classes).to(device)
        class_total = torch.zeros(num_classes).to(device)

        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validating'):
                points = batch['points'].to(device, non_blocking=True)
                colors = batch['colors'].to(device, non_blocking=True)
                labels = batch['labels'].to(device, non_blocking=True)
                
                outputs = model(points, colors)
                loss = criterion(outputs, labels)
                
                # 计算准确率
                pred = outputs.max(1)[1]
                acc = pred.eq(labels).float().mean()
                
                # 更新统计
                val_loss.update(loss.item())
                val_acc.update(acc.item())
                
                # 计算每个类别的准确率
                for i in range(num_classes):
                    mask = labels == i
                    class_correct[i] += pred[mask].eq(labels[mask]).sum()
                    class_total[i] += mask.sum()
                
                # 更新混淆矩阵
                for t, p in zip(labels.view(-1), pred.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1

        
        # 计算每个类别的准确率
        class_acc = class_correct / (class_total + 1e-6)
        
        # 在验证循环结束后，class_acc计算之后添加
        #使用矩阵运算而不是循环来计算IoU
        #保持在GPU上计算，减少CPU和GPU之间的数据传输
        # 计算每个类别的IoU
        intersection = torch.diag(confusion_matrix)  # 对角线上的值是正确预测的数量
        union = confusion_matrix.sum(1) + confusion_matrix.sum(0) - torch.diag(confusion_matrix)  # 并集
        iou = intersection / (union + 1e-6)  # 每个类别的IoU
        miou = iou.mean()  # mIoU

        # 记录到日志和tensorboard
        logger.info(f"Mean IoU: {miou.item()*100:.2f}%")
        writer.add_scalar('Metrics/mIoU', miou.item(), epoch)

        # 记录每个类别的IoU
        for i, class_iou in enumerate(iou):
            writer.add_scalar(f'IoU/class_{i}', class_iou.item(), epoch)
            logger.info(f"Class {i} IoU: {class_iou.item()*100:.2f}%")


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


class WeightedCrossEntropyLoss(nn.Module): #new
    def __init__(self, weight=None):
        super().__init__()
        self.weight = weight

    def forward(self, pred, target):
        # 计算每个类别的权重
        if self.weight is None:
            weight = torch.ones(pred.size(1)).to(pred.device)
        else:
            weight = self.weight

        # 应用权重的交叉熵损失
        loss = F.cross_entropy(pred, target, weight=weight)
        return loss


def compute_class_weights(dataset, num_classes=None): #new
    """计算类别权重"""
    class_counts = torch.zeros(num_classes)
    for batch in dataset:
        labels = batch['labels']
        for i in range(num_classes):
            class_counts[i] += (labels == i).sum()

    # 计算权重
    total = class_counts.sum()
    weights = total / (class_counts * num_classes)
    return weights

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


if __name__ == '__main__':
    try:
        train()
    except Exception as e:
        logging.exception("Training failed with exception:")
        raise
