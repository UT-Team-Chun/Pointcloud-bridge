import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path

class BridgePointCloudDataset(Dataset):
    def __init__(self, data_dir, num_points=8192, transform=None):
        """
        Args:
            data_dir: 数据目录路径
            num_points: 采样点数
            transform: 数据增强转换
        """
        self.data_dir = Path(data_dir)
        self.num_points = num_points
        self.transform = transform
        
        # 获取所有las文件路径
        self.file_list = list(self.data_dir.glob("*.las"))
        
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        # 读取las文件
        las_path = self.file_list[idx]
        las_data = laspy.read(las_path)
        
        # 提取点云数据
        points = np.vstack((las_data.x, las_data.y, las_data.z)).T
        colors = np.vstack((las_data.red, las_data.green, las_data.blue)).T / 65535.0
        labels = las_data.classification
        
        # 随机采样
        if len(points) > self.num_points:
            choice = np.random.choice(len(points), self.num_points, replace=False)
            points = points[choice]
            colors = colors[choice]
            labels = labels[choice]
        
        # 数据增强
        if self.transform:
            points, colors = self.transform(points, colors)
        
        # 转换为张量
        points = torch.FloatTensor(points)
        colors = torch.FloatTensor(colors)
        labels = torch.LongTensor(labels)
        
        return {
            'points': points,
            'colors': colors,
            'labels': labels
        }

# 数据增强
class PointCloudTransform:
    def __init__(self, jitter_sigma=0.01, jitter_clip=0.05):
        self.jitter_sigma = jitter_sigma
        self.jitter_clip = jitter_clip

    def __call__(self, points, colors):
        jittered_points = points.copy()
        
        # 添加随机噪声
        noise = np.clip(np.random.normal(0, self.jitter_sigma, points.shape),
                       -self.jitter_clip, self.jitter_clip)
        jittered_points += noise
        
        # 随机旋转（绕z轴）
        theta = np.random.uniform(0, 2*np.pi)
        rotation_matrix = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
        ])
        jittered_points = np.dot(jittered_points, rotation_matrix)
        
        return jittered_points, colors

# 创建数据加载器
def create_dataloader(data_dir, batch_size=16, num_points=8192):
    dataset = BridgePointCloudDataset(
        data_dir=data_dir,
        num_points=num_points,
        transform=PointCloudTransform()
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    return dataloader
