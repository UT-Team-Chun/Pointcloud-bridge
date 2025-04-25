import logging
import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import traceback # 导入 traceback 模块

class SimplePointCloudDataset(Dataset):
    def __init__(self, data_dir, file_list=None, num_points=4096, transform=False, steps_per_file=10, logger=None): # 增加 steps_per_file 参数
        """
        一个简单的数据集类，从HDF5文件中随机采样固定数量的点。

        Args:
            data_dir (str): 数据目录。
            file_list (list, optional): 要使用的特定文件名列表。如果为None，则使用data_dir中的所有.h5文件。
            num_points (int): 每个样本要采样的点数。
            transform (bool): 是否应用数据增强。
            steps_per_file (int): 每个 epoch 中从每个文件采样的次数。
            logger (logging.Logger, optional): 用于日志记录的记录器实例。
        """
        self.data_dir = data_dir
        self.num_points = num_points
        self.transform = transform
        self.steps_per_file = steps_per_file # 存储 steps_per_file
        self.logger = logger if logger else logging.getLogger(__name__)
        # 获取文件列表
        if file_list is None:
            try:
                self.file_paths = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.h5')]) #排序确保一致性
            except FileNotFoundError:
                self.logger.error(f"Data directory not found: {data_dir}")
                self.file_paths = []
        else:
            # 确保 file_list 中的文件存在于 data_dir 中
            self.file_paths = sorted([os.path.join(data_dir, f) for f in file_list if f.endswith('.h5') and os.path.exists(os.path.join(data_dir, f))])


        if not self.file_paths:
            raise ValueError(f"No valid .h5 files found in {data_dir} or provided file list.")

        self.logger.info(f"Initialized SimplePointCloudDataset with {len(self.file_paths)} files. Steps per file: {self.steps_per_file}.")
        self.logger.info(f"Total samples per epoch: {len(self.file_paths) * self.steps_per_file}")
        # for f_path in self.file_paths:
        #      self.logger.debug(f"  - Found file: {os.path.basename(f_path)}") # 减少日志量

    def normalize_points(self, points):
        """将点云坐标归一化到单位球内"""
        centroid = np.mean(points, axis=0)
        points_centered = points - centroid
        # 使用 try-except 捕获潜在的数值错误
        try:
            max_dist = np.max(np.sqrt(np.sum(points_centered ** 2, axis=1)))
            if max_dist > 1e-6: # 避免除以零
                points_normalized = points_centered / max_dist
            else:
                points_normalized = points_centered # 如果点云只有一个点或所有点重合
        except Exception as e:
            self.logger.error(f"Error during normalization: {e}. Points shape: {points.shape}")
            # 返回未归一化的中心化点云作为后备
            points_normalized = points_centered
        return points_normalized

    def apply_transform(self, points, colors):
        """应用数据增强"""
        if not self.transform:
            return points, colors

        # 使用 try-except 捕获潜在的错误
        try:
            points_transformed = points.copy()
            colors_transformed = colors.copy() if colors is not None else None

            # 随机旋转 (仅绕Z轴)
            theta = np.random.uniform(0, 2 * np.pi)
            rotation_matrix = np.array([
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1]
            ], dtype=points.dtype)
            points_transformed = np.dot(points_transformed, rotation_matrix)

            # 随机缩放 (各向同性)
            scale = np.random.uniform(0.9, 1.1)
            points_transformed *= scale

            # 随机平移
            translation = np.random.uniform(-0.05, 0.05, size=(1, 3)).astype(points.dtype)
            points_transformed += translation

            # 随机抖动颜色
            if colors_transformed is not None:
                color_noise = np.random.normal(0, 0.02, colors_transformed.shape).astype(colors.dtype)
                colors_transformed = np.clip(colors_transformed + color_noise, 0, 1)

            return points_transformed, colors_transformed
        except Exception as e:
            self.logger.error(f"Error during transformation: {e}. Points shape: {points.shape}")
            # 返回原始点和颜色作为后备
            return points, colors


    def __len__(self):
        """返回数据集的总样本数 (文件数 * 每个文件的步数)"""
        return len(self.file_paths) * self.steps_per_file

    def __getitem__(self, idx):
        """加载、采样和处理单个样本"""
        # 计算文件索引和在该文件内的采样索引（虽然在这个实现中后者没用到，但逻辑上是这样）
        file_idx = idx // self.steps_per_file
        # sample_in_file_idx = idx % self.steps_per_file # 当前未使用

        if file_idx >= len(self.file_paths):
             self.logger.error(f"Index {idx} out of bounds. Max file index: {len(self.file_paths)-1}")
             # 返回占位符
             return self._get_placeholder_item("Index out of bounds")

        file_path = self.file_paths[file_idx]
        filename = os.path.basename(file_path)
        # self.logger.debug(f"Loading item {idx} (File {file_idx}, Sample {sample_in_file_idx}): {filename}")

        try:
            # --- 文件加载 ---
            try:
                with h5py.File(file_path, 'r') as f:
                    points_orig = np.array(f['points'])
                    # 检查是否存在 'colors' 和 'labels' 数据集
                    colors_orig = np.array(f['colors']) if 'colors' in f else np.zeros_like(points_orig) # 提供默认值
                    labels_orig = np.array(f['labels']) if 'labels' in f else np.zeros(points_orig.shape[0], dtype=np.int64) # 提供默认值
            except Exception as e:
                self.logger.error(f"Error loading file {filename} (idx: {idx}): {e}")
                self.logger.error(traceback.format_exc()) # 打印详细的回溯信息
                return self._get_placeholder_item(filename)

            num_available_points = points_orig.shape[0]

            if num_available_points == 0:
                 self.logger.warning(f"File {filename} (idx: {idx}) contains 0 points.")
                 return self._get_placeholder_item(filename)

            # --- 采样逻辑 ---
            try:
                if num_available_points >= self.num_points:
                    selected_indices = np.random.choice(num_available_points, self.num_points, replace=False)
                else:
                    indices_all = np.arange(num_available_points)
                    indices_repeat = np.random.choice(num_available_points, self.num_points - num_available_points, replace=True)
                    selected_indices = np.concatenate((indices_all, indices_repeat))
                    np.random.shuffle(selected_indices)
            except Exception as e:
                 self.logger.error(f"Error during sampling for file {filename} (idx: {idx}): {e}")
                 self.logger.error(traceback.format_exc())
                 return self._get_placeholder_item(filename)


            # --- 数据提取和处理 ---
            try:
                sampled_points = points_orig[selected_indices]
                sampled_colors = colors_orig[selected_indices]
                sampled_labels = labels_orig[selected_indices]

                # 归一化坐标
                normalized_points = self.normalize_points(sampled_points)

                # 应用数据增强
                transformed_points, transformed_colors = self.apply_transform(normalized_points, sampled_colors)

                # 检查 NaN 或 Inf
                if not np.all(np.isfinite(transformed_points)):
                    self.logger.warning(f"NaN or Inf detected in transformed points for file {filename} (idx: {idx}).")
                    # 可以选择返回占位符或尝试修复
                    transformed_points = np.nan_to_num(transformed_points) # 尝试修复

                if transformed_colors is not None and not np.all(np.isfinite(transformed_colors)):
                     self.logger.warning(f"NaN or Inf detected in transformed colors for file {filename} (idx: {idx}).")
                     transformed_colors = np.nan_to_num(transformed_colors) # 尝试修复


            except Exception as e:
                 self.logger.error(f"Error during data processing/transform for file {filename} (idx: {idx}): {e}")
                 self.logger.error(traceback.format_exc())
                 return self._get_placeholder_item(filename)


            return {
                'points': transformed_points.astype(np.float32),
                'colors': transformed_colors.astype(np.float32),
                'labels': sampled_labels.astype(np.int64),
                'original_points': sampled_points.astype(np.float32),
                'original_colors': sampled_colors.astype(np.float32),
                'file_name': filename,
                'indices': selected_indices.astype(np.int64)
            }

        except Exception as e:
            # 捕获 __getitem__ 中任何其他未预料到的错误
            self.logger.error(f"Unexpected error in __getitem__ for index {idx}, file {filename}: {e}")
            self.logger.error(traceback.format_exc())
            return self._get_placeholder_item(filename)

    def _get_placeholder_item(self, filename="unknown"):
        """返回一个占位符样本，用于错误处理"""
        self.logger.warning(f"Returning placeholder item for file: {filename}")
        return {
            'points': np.zeros((self.num_points, 3), dtype=np.float32),
            'colors': np.zeros((self.num_points, 3), dtype=np.float32),
            'labels': np.zeros((self.num_points,), dtype=np.int64),
            'original_points': np.zeros((self.num_points, 3), dtype=np.float32),
            'original_colors': np.zeros((self.num_points, 3), dtype=np.float32),
            'file_name': f"error_{filename}",
            'indices': np.arange(self.num_points, dtype=np.int64)
        }


# 使用示例
if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    # 假设你的数据在 'd:/Work/Pointcloud-WL/Pointcloud-bridge/Highway_bridge/data/CB/all/train'
    # 请根据你的实际路径修改
    data_directory = '../../data/CB/all/train' # 示例路径，请修改
    val_directory = '../../data/CB/all/val'   # 示例路径，请修改

    try:
        # 初始化数据集
        dataset = SimplePointCloudDataset(
            data_dir=data_directory,
            num_points=4096,
            transform=True, # 启用训练时的数据增强
            steps_per_file=50, # 每个文件每个epoch采样50次
            logger=logger
        )
        val_dataset = SimplePointCloudDataset(
            data_dir=val_directory,
            num_points=4096,
            transform=False,
            steps_per_file=10, # 验证时可以少采样几次
            logger=logger
        )


        # 检查数据集大小
        logger.info(f"Train dataset size: {len(dataset)}")
        logger.info(f"Val dataset size: {len(val_dataset)}")


        if len(dataset) > 0:
            # 创建DataLoader
            dataloader = DataLoader(
                dataset,
                batch_size=4, # 使用较小的batch size进行测试
                shuffle=True,
                num_workers=0, # !!! 设置为 0 进行调试 !!!
                pin_memory=False # 禁用 pin_memory 进一步简化调试
            )

            # 测试加载一个批次
            logger.info("Loading one batch...")
            batch_count = 0
            for i, batch in enumerate(dataloader):
                batch_count += 1
                if i % 10 == 0: # 每10个batch打印一次日志
                    logger.info(f"Batch {i+1} loaded.")
                    logger.info(f"  Points shape: {batch['points'].shape}, dtype: {batch['points'].dtype}")
                    # ... (其他日志信息) ...
                    logger.info(f"  File names in batch: {batch['file_name']}")

                # 可以在这里添加更多的检查，例如检查NaN
                if torch.isnan(batch['points']).any():
                    logger.error(f"NaN detected in points for batch {i+1}, files: {batch['file_name']}")
                if torch.isnan(batch['colors']).any():
                    logger.error(f"NaN detected in colors for batch {i+1}, files: {batch['file_name']}")

                if batch_count >= 50: # 测试加载50个批次
                     logger.info("Successfully loaded 50 batches.")
                     break
            logger.info(f"Finished testing dataloader after {batch_count} batches.")
        else:
            logger.warning("Dataset is empty, cannot test DataLoader.")

    except ValueError as e:
        logger.error(f"Error initializing dataset: {e}")
        logger.error(traceback.format_exc())
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        logger.error(traceback.format_exc())

