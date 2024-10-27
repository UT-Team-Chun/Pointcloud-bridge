# utils/prepare_data.py
import os
import shutil
import random
from pathlib import Path

def split_dataset(data_dir, train_ratio=0.7, val_ratio=0.15):
    """
    划分数据集
    """
    data_dir = Path(data_dir)
    all_files = list(data_dir.glob('*.las'))
    random.shuffle(all_files)
    
    n_total = len(all_files)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    train_files = all_files[:n_train]
    val_files = all_files[n_train:n_train+n_val]
    test_files = all_files[n_train+n_val:]
    
    # 移动文件到对应目录
    for files, subset in [(train_files, 'train'), 
                         (val_files, 'val'), 
                         (test_files, 'test')]:
        subset_dir = data_dir.parent / subset
        subset_dir.mkdir(exist_ok=True)
        for f in files:
            shutil.copy2(f, subset_dir / f.name)

if __name__ == '__main__':
    split_dataset('../downsamoled_data')
