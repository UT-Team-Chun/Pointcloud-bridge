import logging
import os
from pathlib import Path

import laspy
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from tqdm import tqdm


def setup_logging(log_dir):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir +'/testing.log', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def create_new_las_file(points, colors, labels, output_path):
    """创建新的las文件，只包含xyz, rgb和classification"""
    # 创建LAS文件
    las = laspy.create(file_version="1.3", point_format=3)

    # 写入点数据
    las.x = points[:, 0]
    las.y = points[:, 1]
    las.z = points[:, 2]

    # 设置头部信息
    #las.header.offsets = [np.min(las.x), np.min(las.y), np.min(las.z)]
    #las.header.scales = [0.001, 0.001, 0.001]
    
    # 设置RGB值 (需要从0-1范围转换到0-65535范围)
    las.red = (colors[:, 0] * 65535).astype(np.uint16)
    las.green = (colors[:, 1] * 65535).astype(np.uint16)
    las.blue = (colors[:, 2] * 65535).astype(np.uint16)
    
    # 设置分类标签
    las.classification = labels.astype(np.uint8)
    
    # 写入文件
    las.write(output_path)


# 定义评价指标计算函数
def calculate_metrics(confusion_matrix):
    """
    从混淆矩阵计算所有指标
    """
    cm = confusion_matrix.cpu().numpy()

    # 计算每个类别的IoU
    intersection = np.diag(cm)
    union = np.sum(cm, axis=1) + np.sum(cm, axis=0) - np.diag(cm)
    iou_per_class = intersection / (union + 1e-6)
    miou = np.nanmean(iou_per_class)

    # 计算OA (Overall Accuracy)
    oa = np.sum(np.diag(cm)) / np.sum(cm)

    # 计算每个类别的准确率
    acc_per_class = np.diag(cm) / (np.sum(cm, axis=1) + 1e-6)
    macc = np.nanmean(acc_per_class)  # Mean Accuracy

    # 计算每个类的precision和recall
    precision_per_class = np.diag(cm) / (np.sum(cm, axis=0) + 1e-6)
    recall_per_class = np.diag(cm) / (np.sum(cm, axis=1) + 1e-6)

    # 计算加权平均的precision和recall
    weights = np.sum(cm, axis=1) / np.sum(cm)
    precision = np.sum(precision_per_class * weights)
    recall = np.sum(recall_per_class * weights)

    # 计算F1 score
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)

    return {
        'mIoU': miou,
        'IoU_per_class': iou_per_class,
        'OA': oa,
        'mAcc': macc,
        'Acc_per_class': acc_per_class,
        'Precision': precision,
        'Recall': recall,
        'F1_score': f1,
        'Confusion_Matrix': cm
    }



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

def main():
    # 配置参数
    config = {
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

    exp_dir= 'experiments/exp_122920_brimulti_brienc_CB_all_weightedloss'
    checkpoint_path = os.path.join(exp_dir, 'best_model.pth')
    all_true_labels = []
    all_predictions = []
    logger = setup_logging(exp_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')

    from experiments.exp_122920_brimulti_brienc_CB_all_weightedloss.utils.BriPCDMulti import BriPCDMulti
    from experiments.exp_122920_brimulti_brienc_CB_all_weightedloss.models.model import EnhancedPointNet2
    from torch.utils.data import DataLoader

    test_dataset = BriPCDMulti(
        data_dir='../data/CB/all/val',
        num_points=config['num_points'],
        block_size=2.0,
        sample_rate=0.4,
        logger=logger
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    

    #from models.enhanced_pointnet2 import EnhancedPointNet2
    # 加载模型

    model = EnhancedPointNet2(config['num_classes']).to(device)

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f'Loaded model from {checkpoint_path}')
    else:
        logger.error(f'No checkpoint found at {checkpoint_path}')
        return
    
    model.eval()
    
    # 创建输出目录
    output_dir=Path(exp_dir) / 'predicted_las'
    output_dir.mkdir(parents=True, exist_ok=True)

    val_acc = AverageMeter()
    confusion_matrix = torch.zeros(config['num_classes'], config['num_classes']).to(device)
    class_correct = torch.zeros(config['num_classes']).to(device)
    class_total = torch.zeros(config['num_classes']).to(device)

    # 创建一个字典来存储每个文件的预测结果
    file_predictions = {}

    with torch.no_grad():
        all_preds = []
        all_labels = []

        #pbar=tqdm(test_files, desc='Processing files', position=0, leave=True)
        pbar=tqdm(test_loader,total=len(test_loader),desc='Processing files', position=0, leave=True)

        for batch in pbar:

            points = batch['points'].to(device, non_blocking=True)
            colors = batch['colors'].to(device, non_blocking=True)
            labels = batch['labels'].to(device, non_blocking=True)
            file_names = batch['file_name']  # 获取文件名
            original_points = batch['original_points']
            original_colors = batch['original_colors']
            indices = batch['indices']

            outputs = model(points, colors)

            # 计算准确率
            pred = outputs.max(1)[1]
            acc_1 = pred.eq(labels).float().mean()

            # 更新统计
            val_acc.update(acc_1.item())

            all_preds.append(pred)
            all_labels.append(labels)

            # 将预测结果存储到对应文件
            for i in range(len(file_names)):
                fname = file_names[i]
                if fname not in file_predictions:
                    file_predictions[fname] = {
                        'points': [],
                        'colors': [],
                        'predictions': [],
                        'indices': []
                    }

                file_predictions[fname]['points'].append(original_points[i])
                file_predictions[fname]['colors'].append(original_colors[i])
                file_predictions[fname]['predictions'].append(pred[i].cpu())
                file_predictions[fname]['indices'].append(indices[i])

            # 更新进度条
            pbar.set_postfix({'Val_Acc': f'{val_acc.avg * 100:.2f}%'})


        all_preds = torch.cat(all_preds, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        # 计算总体准确率
        total_acc = all_preds.eq(all_labels).float().mean()
        # val_acc.update(total_acc.item())

        # accuracy for each class
        for i in range(config['num_classes']):
            mask = all_labels == i
            class_correct[i] = all_preds[mask].eq(all_labels[mask]).sum()
            class_total[i] = mask.sum()

        # 计算每个类别的准确率
        class_acc = class_correct / (class_total + 1e-6)

        # confusion matrix

        confusion_matrix = torch.zeros(config['num_classes'], config['num_classes'], dtype=torch.long)
        for t, p in zip(all_labels.view(-1), all_preds.view(-1)):
            confusion_matrix[t.long(), p.long()] += 1

        # 计算所有指标
        metrics = calculate_metrics(confusion_matrix)

        # 记录结果
        logger.info(f"Mean IoU: {metrics['mIoU'] * 100:.2f}%")
        logger.info(f"Overall Accuracy: {metrics['OA'] * 100:.2f}%")
        logger.info(f"Mean Accuracy: {metrics['mAcc'] * 100:.2f}%")
        logger.info(f"Precision: {metrics['Precision'] * 100:.2f}%")
        logger.info(f"Recall: {metrics['Recall'] * 100:.2f}%")
        logger.info(f"F1 Score: {metrics['F1_score'] * 100:.2f}%")

        # 记录每个类别的IoU
        for i, class_iou in enumerate(metrics['IoU_per_class']):
            logger.info(f"Class {i} IoU: {class_iou * 100:.2f}%")


        # # 处理并保存预测结果
        # for fname, data in file_predictions.items():
        #     # 将所有块的数据合并
        #     all_points = np.concatenate(data['points'], axis=0)
        #     all_colors = np.concatenate(data['colors'], axis=0)
        #     all_predictions = np.concatenate(data['predictions'], axis=0)
        #     all_indices = np.concatenate(data['indices'], axis=0)
        #
        #     # 创建新的las文件
        #     new_las = laspy.create(point_format=3)  # 使用点格式3，包含坐标和颜色
        #
        #     # 设置点云数据
        #     new_las.x = all_points[:, 0]
        #     new_las.y = all_points[:, 1]
        #     new_las.z = all_points[:, 2]
        #
        #     # 设置颜色数据（需要转换回0-65535范围）
        #     new_las.red = (all_colors[:, 0] * 65535).astype(np.uint16)
        #     new_las.green = (all_colors[:, 1] * 65535).astype(np.uint16)
        #     new_las.blue = (all_colors[:, 2] * 65535).astype(np.uint16)
        #
        #     # 设置分类标签
        #     new_las.classification = all_predictions
        #
        #     # 保存文件
        #     output_path = output_dir / f'predicted_{fname}'
        #     new_las.write(output_path)
        #
        #     logger.info(f'Saved predicted result to {output_path}')

        # 在main函数中调用可视化
        fig = visualize_results(metrics, class_names=['Background', 'Abundant','Girder', 'Deck', 'parapet'])
        plt.savefig(os.path.join(exp_dir, 'evaluation_results.png'), dpi=300, bbox_inches='tight')
        plt.close()


def test():
    # 设置随机种子保证可复现性
    torch.manual_seed(42)

    # 创建测试数据
    batch_size = 2
    num_points = 1024
    num_classes = 4
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 创建随机输入数据
    xyz = torch.randn(batch_size, num_points, 3)
    features = torch.randn(batch_size, num_points, 3)

    # from models.enhanced_pointnet2 import EnhancedPointNet2
    # 加载模型
    from experiments.exp_122920_brimulti_brienc_CB_all_weightedloss.models.enhanced_pointnet2 import EnhancedPointNet2
    model = EnhancedPointNet2(num_classes)
    checkpoint_path = 'experiments/exp_122920_brimulti_brienc_CB_all_weightedloss/latest_checkpoint.pth'

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f'Loaded model from {checkpoint_path}')
    else:
        print(f'No checkpoint found at {checkpoint_path}')
        return

    model.eval()

    # 1. 基础功能测试
    print("=" * 50)
    print("基础功能测试")
    print(f"输入 xyz shape: {xyz.shape}")
    print(f"输入 features shape: {features.shape}")

    try:
        output = model(xyz, features)
        print(f"输出 shape: {output.shape}")
        print("模型前向传播测试通过!")
    except Exception as e:
        print(f"模型运行出错: {str(e)}")


def visualize_results(metrics, class_names=None):
    """
    可视化评估结果
    """
    if class_names is None:
        class_names = [f'Class {i}' for i in range(len(metrics['IoU_per_class']))]

    # 设置风格
    #plt.style.use('seaborn')

    # 创建一个3x2的子图布局
    fig = plt.figure(figsize=(20, 15))

    # 1. 混淆矩阵可视化
    ax1 = plt.subplot(231)
    cm = metrics['Confusion_Matrix']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Confusion Matrix', pad=20)
    plt.xlabel('Predicted')
    plt.ylabel('True')

    # 2. 每个类别的IoU柱状图
    ax2 = plt.subplot(232)
    iou_data = metrics['IoU_per_class'] * 100
    sns.barplot(x=class_names, y=iou_data)
    plt.title(f'IoU per Class (mIoU: {metrics["mIoU"] * 100:.2f}%)')
    plt.xticks(rotation=45)
    plt.ylabel('IoU (%)')

    # 3. 每个类别的准确率柱状图
    ax3 = plt.subplot(233)
    acc_data = metrics['Acc_per_class'] * 100
    sns.barplot(x=class_names, y=acc_data)
    plt.title(f'Accuracy per Class (mAcc: {metrics["mAcc"] * 100:.2f}%)')
    plt.xticks(rotation=45)
    plt.ylabel('Accuracy (%)')

    # 4. 归一化的混淆矩阵
    ax4 = plt.subplot(234)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='RdYlBu_r',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Normalized Confusion Matrix', pad=20)
    plt.xlabel('Predicted')
    plt.ylabel('True')

    # 5. 总体指标比较
    ax5 = plt.subplot(235)
    metrics_names = ['OA', 'mAcc', 'mIoU', 'F1_score']
    metrics_values = [metrics['OA'], metrics['mAcc'],
                      metrics['mIoU'], metrics['F1_score']]
    sns.barplot(x=metrics_names, y=[x * 100 for x in metrics_values])
    plt.title('Overall Metrics Comparison')
    plt.ylabel('Percentage (%)')

    plt.tight_layout()
    return fig




if __name__ == '__main__':
    main()
    #test()
