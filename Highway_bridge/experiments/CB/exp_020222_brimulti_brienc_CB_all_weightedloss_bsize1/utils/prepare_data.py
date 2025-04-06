# utils/prepare_data.py
import os
import shutil
import random
from pathlib import Path

def split_dataset(data_dir, train_ratio=0.7, val_ratio=0.15):
    """
    划分数据集
    
    Args:
        data_dir (str or Path): 原始数据目录的路径
        train_ratio (float): 训练集比例
        val_ratio (float): 验证集比例
    """
    # 确保data_dir是Path对象
    data_dir = Path(data_dir)
    
    # 确保输入目录存在
    if not data_dir.exists():
        raise ValueError(f"数据目录不存在: {data_dir}")
    
    # 获取所有.las文件
    all_files = list(data_dir.glob('*.las'))
    if not all_files:
        raise ValueError(f"在 {data_dir} 中没有找到.las文件")
    
    print(f"找到总文件数: {len(all_files)}")
    
    # 随机打乱文件列表
    random.shuffle(all_files)
    
    # 计算划分数量
    n_total = len(all_files)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    # 划分文件列表
    train_files = all_files[:n_train]
    val_files = all_files[n_train:n_train+n_val]
    test_files = all_files[n_train+n_val:]
    
    # 创建数据集根目录
    dataset_root = data_dir.parent
    
    # 移动文件到对应目录
    for files, subset in [(train_files, 'train'), 
                         (val_files, 'val'), 
                         (test_files, 'test')]:
        # 创建子集目录
        subset_dir = dataset_root / subset
        subset_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n处理 {subset} 集:")
        print(f"目标目录: {subset_dir}")
        print(f"文件数量: {len(files)}")
        
        # 复制文件
        for f in files:
            try:
                dest_path = subset_dir / f.name
                shutil.copy2(str(f), str(dest_path))
                print(f"成功复制: {f.name}")
            except Exception as e:
                print(f"复制文件 {f} 时出错: {str(e)}")

if __name__ == '__main__':
    try:
        # 使用绝对路径
        current_dir = Path(__file__).parent.parent  # 获取当前文件的父目录的父目录
        data_dir = current_dir / 'data' / 'downsampled_data'
        
        print(f"数据目录: {data_dir}")
        
        # 确保目录存在
        if not data_dir.exists():
            raise ValueError(f"数据目录不存在: {data_dir}")
        
        split_dataset(data_dir)
        print("\n数据集划分完成！")
        
    except Exception as e:
        print(f"发生错误: {str(e)}")
