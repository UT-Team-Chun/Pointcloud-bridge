# train.py
import datetime
import logging
import shutil
# tensorboard --logdir ./logs
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# from models.model import EnhancedPointNet2
from models.model import EnhancedPointNet2
from utils.BriPCDMulti_comp import BriPCDMulti
from utils.logger_config import initialize_logger, get_logger

# 配置参数

config = {
    'transformer_config': {
        'trans_dim': 384,
        'depth': 12,
        'drop_path_rate': 0.1,
        'num_heads': 6,
        'encoder_dims': 384,
    },
    'num_group': 64,  # 移到外层
    'group_size': 32,  # 移到外层
    'num_points': 4096,
    'chunk_size': 4096,
    'overlap': 1024,
    'batch_size': 16,
    'num_workers': 0,
    'learning_rate': 0.001,
    'num_classes': 5,
    'num_epochs': 500,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

#{noise:0, abutment:1, girder:2, slab:3, parapet:4}

def train():
    # 创建实验目录
    timestamp = datetime.datetime.now().strftime('%m%d%H')
    case = 'brimulti_brienc_CB_all_weightedloss'
    exp_dir = Path(f'experiments/exp_{timestamp}_{case}')
    exp_dir.mkdir(parents=True, exist_ok=True)

    # wandb初始化
    wandb.init(
        project="bridge-segmentation",  # 你的项目名称
        name=f"{timestamp}_{case}",     # 实验名称
        config=config                    # 配置参数
    )

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
    train_dataset = BriPCDMulti(
        data_dir='../data/CB/all-2/train/',
        num_points=4096,
        block_size=2.0,
        sample_rate=0.4,
        logger=get_logger(),
        transform=True
    )

    val_dataset = BriPCDMulti(
        data_dir='../data/CB/all-2/val/',
        num_points=4096,
        block_size=2.0,
        sample_rate=0.4,
        logger=get_logger()
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

    #model = EnhancedPointNet2(config, num_classes=5).to(device)
    model = EnhancedPointNet2(num_classes).to(device)
    #model = PointNet2(num_classes).to(device)
    wandb.watch(model)
    source_path = Path('models')
    destination_path = exp_dir / 'models'
    utils_path = Path('utils')
    shutil.copytree(source_path, destination_path)
    shutil.copytree(utils_path, exp_dir / 'utils')
    shutil.copy2('train_MulSca_BriStruNet_CB.py', exp_dir/'train_MulSca_BriStruNet_CB.py')

    # 损失函数和优化器
    #criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], betas=(0.9, 0.999),
                           weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',  # 监控指标是越大越好
        factor=0.1,  # 学习率调整因子
        patience=5  # 容忍多少个epoch指标没改善
    )

    # 训练循环
    best_val_loss = float('inf')
    best_val_acc = 0.0

    # 计算类别权重
    class_weights = compute_class_weights(train_loader.dataset,num_classes)
    criterion = WeightedCrossEntropyLoss(weight=class_weights.to(device))
    #criterion = CombinedLoss(alpha= 0.8 ).to(device)
    #criterion = DiceLoss(smooth= 1e-5 ).to(device)

    for epoch in range(1, config['num_epochs']+1):
        # 训练阶段
        model.train()
        train_loss = AverageMeter()
        train_acc = AverageMeter()

        # 创建进度条
        total_samples = len(train_dataset)
        logger.info(f'Total samples is: {train_loader}')
        pbar = tqdm(train_loader, desc=f'Epoch [{epoch}/{config["num_epochs"]}] Training',
                    total=total_samples // config['batch_size'], position=0, leave=True)

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
                'Acc': f'{train_acc.avg * 100:.2f}%'
            })


        # 记录学习率
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('Learning_rate', current_lr, epoch)

        # Validation phase
        model.eval()
        val_loss = AverageMeter()
        val_acc = AverageMeter()
        confusion_matrix = torch.zeros(num_classes, num_classes).to(device)  # 混淆矩阵
        class_correct = torch.zeros(num_classes).to(device)
        class_total = torch.zeros(num_classes).to(device)

        with torch.no_grad():
            all_preds = []
            all_labels = []
            total_samples = len(val_loader)
            pbar = tqdm(val_loader, desc=f'Epoch [{epoch}/{config["num_epochs"]}] Validating',
                        total=total_samples // config['batch_size'], position=0, leave=True)
            for batch in pbar:
                points = batch['points'].to(device, non_blocking=True)
                colors = batch['colors'].to(device, non_blocking=True)
                labels = batch['labels'].to(device, non_blocking=True)

                outputs = model(points, colors)
                loss = criterion(outputs, labels)

                # 计算准确率
                pred = outputs.max(1)[1]
                acc_1 = pred.eq(labels).float().mean()

                # 更新统计
                val_loss.update(loss.item())
                val_acc.update(acc_1.item())

                all_preds.append(pred)
                all_labels.append(labels)

                # 更新进度条
                pbar.set_postfix({
                    'Val_Loss': f'{val_loss.avg:.4f}',
                    'Val_Acc': f'{val_acc.avg * 100:.2f}%'
                })

        start_time = datetime.datetime.now()

        all_preds = torch.cat(all_preds, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        # 计算总体准确率
        total_acc = all_preds.eq(all_labels).float().mean()
        #val_acc.update(total_acc.item())

        # 更新学习率
        scheduler.step(total_acc)

        # accuracy for each class
        for i in range(num_classes):
            mask = all_labels == i
            class_correct[i] = all_preds[mask].eq(all_labels[mask]).sum()
            class_total[i] = mask.sum()

        # 计算每个类别的准确率
        class_acc = class_correct / (class_total + 1e-6)

        # confusion matrix

        # confusion_matrix = torch.zeros(num_classes, num_classes, dtype=torch.long)
        # for t, p in zip(all_labels.view(-1), all_preds.view(-1)):
        #     confusion_matrix[t.long(), p.long()] += 1
        #
        # # 计算每个类别的IoU
        # intersection = torch.diag(confusion_matrix)  # 对角线上的值是正确预测的数量
        # union = confusion_matrix.sum(1) + confusion_matrix.sum(0) - torch.diag(confusion_matrix)  # 并集
        # iou = intersection / (union + 1e-6)  # 每个类别的IoU
        # miou = iou.mean()  # mIoU
        #
        # # 记录到日志和tensorboard
        # logger.info(f"Mean IoU: {miou.item() * 100:.2f}%")
        # writer.add_scalar('Metrics/mIoU', miou.item(), epoch)

        # # 记录每个类别的IoU
        # for i, class_iou in enumerate(iou):
        #     writer.add_scalar(f'IoU/class_{i}', class_iou.item(), epoch)
        #     logger.info(f"Class {i} IoU: {class_iou.item() * 100:.2f}%")

        end_time = datetime.datetime.now()
        logger.info(f"Validation parameter calculation time: {end_time - start_time}")

        # 记录训练信息
        logger.info(f"\nEpoch {epoch}/{config['num_epochs']}:")
        logger.info(f"Train Loss: {train_loss.avg:.4f}, Train Acc: {train_acc.avg * 100:.2f}%")
        logger.info(f"Val Loss: {val_loss.avg:.4f}, Val Acc: {val_acc.avg * 100:.2f}%")

        # 记录到tensorboard
        writer.add_scalar('Loss/train', train_loss.avg, epoch)
        writer.add_scalar('Loss/val', val_loss.avg, epoch)
        writer.add_scalar('Accuracy/train', train_acc.avg, epoch)
        writer.add_scalar('Accuracy/val', val_acc.avg, epoch)

        # wandb记录训练指标
        wandb.log({
            'epoch': epoch,
            'train/loss': train_loss.avg,
            'train/accuracy': train_acc.avg,
            'val/loss': val_loss.avg,
            'val/accuracy': val_acc.avg,
            'learning_rate': current_lr,
        })

        # 记录每个类别的准确率
        for i, acc in enumerate(class_acc):
            writer.add_scalar(f'Class_Accuracy/class_{i}', acc.item(), epoch)
            logger.info(f"Class {i} Accuracy: {acc.item() * 100:.2f}%")
            wandb.log({f'class_accuracy/class_{i}': acc.item()}, step=epoch)

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
            logger.info(f"Saved best model with accuracy: {best_val_acc * 100:.2f}%")

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
        logger.info(f"Saved checkpoint at epoch {epoch}")

    writer.close()
    logger.info("Training completed!")


class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, weight=None):
        super().__init__()
        self.weight = weight

    def forward(self, pred, target):
        if self.weight is None:
            weight = torch.ones(pred.size(1)).to(pred.device)
        else:
            weight = self.weight.to(pred.device)  # 确保权重在正确的设备上
        return F.cross_entropy(pred, target, weight=weight)


def compute_class_weights(dataset, num_classes=None):
    """计算类别权重"""
    class_counts = torch.zeros(num_classes)
    for batch in dataset:
        labels = batch['labels']
        for i in range(num_classes):
            class_counts[i] += (labels == i).sum()

    # 添加小值避免除零
    class_counts = class_counts + 1e-6

    # 计算权重
    total = class_counts.sum()
    weights = total / (class_counts * num_classes)

    # 可选：限制权重范围，避免极端值
    weights = torch.clamp(weights, min=0.5, max=5.0)
    logging.info(f"Class weights: {weights}")

    return weights


import torch
import torch.nn.functional as F



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
