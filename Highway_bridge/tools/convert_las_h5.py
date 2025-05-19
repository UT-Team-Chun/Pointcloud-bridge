import os

import h5py
import laspy
import numpy as np


def convert_las_to_hdf5(las_filename, hdf5_filename):
    """将 LAS 文件转换为 HDF5 文件"""
    # 读取 LAS 文件
    las = laspy.read(las_filename)

    # 获取点坐标
    points = np.vstack((las.x, las.y, las.z)).transpose()

    # 获取颜色信息（如果存在）
    if hasattr(las, 'red') and hasattr(las, 'green') and hasattr(las, 'blue'):
        colors = np.vstack((las.red, las.green, las.blue)).transpose() / 65535.0
    else:
        colors = np.ones_like(points)

    # 获取分类标签（如果存在）
    if hasattr(las, 'classification'):
        labels = np.array(las.classification)
    else:
        labels = np.zeros(len(points), dtype=np.int64)

    # 创建 HDF5 文件并写入数据
    with h5py.File(hdf5_filename, 'w') as f:
        f.create_dataset('points', data=points, compression='gzip')
        f.create_dataset('colors', data=colors, compression='gzip')
        f.create_dataset('labels', data=labels, compression='gzip')

    print(f"Successfully converted {las_filename} to {hdf5_filename}")

# 将所有 LAS 文件转换为 HDF5 文件
if __name__ == '__main__':
    las_dir = 'data/YBC/train'
    hdf5_dir = las_dir

    for las_filename in os.listdir(las_dir):
        if las_filename.endswith('.las'):
            hdf5_filename = os.path.join(hdf5_dir, las_filename.replace('.las', '.h5'))
            convert_las_to_hdf5(os.path.join(las_dir, las_filename), hdf5_filename)
