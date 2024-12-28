import os

import laspy
import numpy as np
from torch.utils.data import Dataset


def read_las_file(las_path):
    """
    读取las文件并返回与原格式相同的数据 (N×7的数组，包含xyzrgb和label)
    """
    # 读取las文件
    las = laspy.read(las_path)
    
    # 获取xyz坐标
    x = las.x
    y = las.y
    z = las.z
    
    # 获取RGB值 (las文件中通常RGB值范围是0-65535，需要转换到0-255)
    if hasattr(las, 'red') and hasattr(las, 'green') and hasattr(las, 'blue'):
        r = las.red / 65535 * 255
        g = las.green / 65535 * 255
        b = las.blue / 65535 * 255
    else:
        # 如果没有颜色信息，设置默认值
        r = np.zeros_like(x)
        g = np.zeros_like(x)
        b = np.zeros_like(x)
    
    # 获取分类标签 (如果存在)
    if hasattr(las, 'classification'):
        labels = las.classification
    else:
        # 如果没有标签，设置默认值0
        labels = np.zeros_like(x)
    
    # 组合所有数据
    points = np.column_stack((x, y, z, r, g, b))
    bridge_data = np.column_stack((points, labels))
    
    return bridge_data

# output.shape: [4096, 9]
class LWBridgeDataset(Dataset):
    def __init__(self, split='train', data_root='trainval_fullarea', num_point=4096, block_size=1.0, sample_rate=1.0, num_class=4, transform=None):
        super().__init__()
        self.num_point = num_point
        self.block_size = block_size
        self.transform = transform

        data_folder = os.path.join(data_root, split)
        bridges = sorted(os.listdir(data_folder))

        self.bridge_points, self.bridge_labels = [], []
        self.bridge_coord_min, self.bridge_coord_max = [], []
        num_point_all = []
        
        # labelweights = np.zeros(num_class)
        # to avoid 0
        labelweights = np.ones(num_class)

        for bridge in bridges:
            bridge_path = os.path.join(data_folder, bridge)
            bridge_data = read_las_file(bridge_path)  # xyzrgbl, N * 7
            points, labels = bridge_data[:, 0: 6], bridge_data[:, 6]  # points (xyzrgb): N * 6; labels(l): N
            tmp, _ = np.histogram(labels, range(num_class + 1))
            # every pair of correspondent elements in 'tmp' and '_' is like a key-value, indicating the number of points in each class, i <= x < i+1
            # tmp.shape: (num_class,)
            # _: [0, 1, 2, ..., num_class]
            labelweights += tmp
            coord_min, coord_max = np.amin(points, axis=0)[: 3], np.amax(points, axis=0)[: 3]
            self.bridge_points.append(points), self.bridge_labels.append(labels)
            self.bridge_coord_min.append(coord_min), self.bridge_coord_max.append(coord_max)
            num_point_all.append(labels.size)
        # class with larger point_num --> easier to train --> needs lower weight
        labelweights = labelweights.astype(np.float32)

        # for WCEL
        self.num_per_cls = labelweights
        
        labelweights = labelweights / np.sum(labelweights) # relative weights whose sum is 1

        # the point_numbers of some classes are relatively few, use 'np.power' for more balanced result
        # make 'labelweights' denominator for inversely proportion so that class with larger point_num has lower weight
        # self.labelweights = np.power(np.amax(labelweights) / labelweights, 1 / 3.0)

        self.labelweights = labelweights

        print('labelweights of ' + split + ': ' + str(self.labelweights))
        sample_prob = num_point_all / np.sum(num_point_all) # point_number of each bridge / the total point_number, (probability)   
        num_iter = int(np.sum(num_point_all) * sample_rate / num_point) # number of sub point clouds (num_iter times to cover all points)
        bridge_idxs = []

        # len(bridges): number of bridges
        for index in range(len(bridges)):
            bridge_idxs.extend([index] * int(round(sample_prob[index] * num_iter)))
            # sample_prob[index]: probability of each bridge to be used
            # sample_prob[index] * num_iter: how many times this bridge is used
            # bridge_idxs = [0 0 0 0 1 1 ...] --> 4 times of bridge[0]
        self.bridge_idxs = np.array(bridge_idxs)
        print("Totally {} samples in {} set.".format(len(self.bridge_idxs), split))

    def __getitem__(self, idx):
        bridge_idx = self.bridge_idxs[idx]
        points = self.bridge_points[bridge_idx] # N * 6
        labels = self.bridge_labels[bridge_idx] # N
        N_points = points.shape[0] # N
        
        try_idx = 0

        while(True):
            center = points[np.random.choice(N_points)][: 3] # randomly select a point as the center (only use coordinates)
            # draw a rectangle
            block_min = center - [self.block_size / 2.0, self.block_size / 2.0, 0]
            block_max = center + [self.block_size / 2.0, self.block_size / 2.0, 0]
            range = block_max - block_min

            # output of np.where: (array of indices, dtype), so only use the first one
            point_idxs = np.where((points[:, 0] >= block_min[0]) &
                                  (points[:, 0] <= block_max[0]) &
                                  (points[:, 1] >= block_min[1]) &
                                  (points[:, 1] <= block_max[1]))[0]
            try_idx += 1

            if (point_idxs.size > 1024) or (try_idx > 100):
                break

        # num_point is bigger than 1024, make the num_point to 4096 anyway
        if point_idxs.size >= self.num_point:
            selected_point_idxs = np.random.choice(point_idxs, self.num_point, replace=False)
        else:
            selected_point_idxs = np.random.choice(point_idxs, self.num_point, replace=True)

        # normalization
        selected_points = points[selected_point_idxs, :]  # num_point * 6
        current_points = np.zeros((self.num_point, 9))  # num_point * 9

        # coordinates normalization
        current_points[:, 6] = selected_points[:, 0] / (self.bridge_coord_max[bridge_idx][0] - self.bridge_coord_min[bridge_idx][0])
        current_points[:, 7] = selected_points[:, 1] / (self.bridge_coord_max[bridge_idx][1] - self.bridge_coord_min[bridge_idx][1])
        current_points[:, 8] = selected_points[:, 2] / (self.bridge_coord_max[bridge_idx][2] - self.bridge_coord_min[bridge_idx][2])

        # move to center (only x and y)
        selected_points[:, 0] = selected_points[:, 0] - center[0]
        selected_points[:, 1] = selected_points[:, 1] - center[1]

        # color (rgb) normalization
        # selected_points[:, 3: 6] /= 255.0
        
        current_points[:, 0: 6] = selected_points
        # current_points.size: num_point * 9, [x (centered), y (centered), z, r~, g~, b~, x~, y~, z~] (~ means normalized)

        current_labels = labels[selected_point_idxs]

        if self.transform is not None:
            current_points, current_labels = self.transform(current_points, current_labels)

        points = current_points[:, : 3].astype(np.float32)
        colors = current_points[:, 3: 6].astype(np.float32)

        return {
            'points': points,
            'colors': colors,
            'labels': current_labels.astype(np.int64),
        }

    def __len__(self):
        return len(self.bridge_idxs)


class ScannetDatasetWholeScene():
    # prepare to give prediction on each points
    def __init__(self, root, block_points=4096, split='test', stride=0.5, block_size=1.0, padding=0.001, num_class=4):
        self.block_points = block_points
        self.block_size = block_size
        self.padding = padding
        self.root = root
        self.split = split
        self.stride = stride
        self.scene_points_num = []

        assert split in ['train', 'test']
        data_folder = os.path.join(root, split)
        self.file_list = [d for d in os.listdir(data_folder)]
        self.scene_points_list = []
        self.semantic_labels_list = []
        self.bridge_coord_min, self.bridge_coord_max = [], []
        for file in self.file_list:
            bridge_path = os.path.join(data_folder, file)
            #data = np.loadtxt(bridge_path)  # xyzrgbl, N * 7
            data = read_las_file(bridge_path)
            points = data[: , : 3]
            self.scene_points_list.append(data[: , : 6]) # xyzrgb
            self.semantic_labels_list.append(data[: , 6]) # label
            coord_min, coord_max = np.amin(points, axis=0)[: 3], np.amax(points, axis=0)[: 3]
            self.bridge_coord_min.append(coord_min), self.bridge_coord_max.append(coord_max)
        assert len(self.scene_points_list) == len(self.semantic_labels_list)

        labelweights = np.zeros(num_class)
        # to avoid 0
        # labelweights = np.ones(num_class)
        for seg in self.semantic_labels_list:
            tmp, _ = np.histogram(seg, range(num_class + 1))
            self.scene_points_num.append(seg.shape[0])
            labelweights += tmp
        # class with larger point_num --> easier to train --> needs lower weight
        labelweights = labelweights.astype(np.float32)
        labelweights = labelweights / np.sum(labelweights)
        # the point_numbers of some classes are relatively few, use 'np.power' for more balanced result
        # make 'labelweights' denominator for inversely proportional so that class with larger point_num has lower weight
        self.labelweights = np.power(np.amax(labelweights) / labelweights, 1 / 3.0)

    def __getitem__(self, index):
        point_set_ini = self.scene_points_list[index]
        points = point_set_ini[: , : 6] # xyzrgb
        labels = self.semantic_labels_list[index] # label
        coord_min, coord_max = np.amin(points, axis=0)[: 3], np.amax(points, axis=0)[: 3]
        # like convolution
        grid_x = int(np.ceil(float(coord_max[0] - coord_min[0] - self.block_size) / self.stride) + 1) # 'ceiling' and 'floor'
        grid_y = int(np.ceil(float(coord_max[1] - coord_min[1] - self.block_size) / self.stride) + 1)
        data_bridge, label_bridge, sample_weight, index_bridge = np.array([]), np.array([]), np.array([]), np.array([])
        for index_y in range(0, grid_y):
            for index_x in range(0, grid_x):
                s_x = coord_min[0] + index_x * self.stride
                e_x = min(s_x + self.block_size, coord_max[0])
                s_x = e_x - self.block_size
                s_y = coord_min[1] + index_y * self.stride
                e_y = min(s_y + self.block_size, coord_max[1])
                s_y = e_y - self.block_size
                point_idxs = np.where((points[:, 0] >= s_x - self.padding) &
                                      (points[:, 0] <= e_x + self.padding) &
                                      (points[:, 1] >= s_y - self.padding) &
                                      (points[:, 1] <= e_y + self.padding))[0]
                if point_idxs.size == 0:
                    continue
                num_batch = int(np.ceil(point_idxs.size / self.block_points))
                point_size = int(num_batch * self.block_points)
                replace = False if (point_size - point_idxs.size <= point_idxs.size) else True
                point_idxs_repeat = np.random.choice(point_idxs, point_size - point_idxs.size, replace=replace)
                point_idxs = np.concatenate((point_idxs, point_idxs_repeat)) # make 'point_idxs.size' the integer multiple of 4096
                np.random.shuffle(point_idxs)
                data_batch = points[point_idxs, : ]

                # coordinates normalization
                normlized_xyz = np.zeros((point_size, 3))
                normlized_xyz[: , 0] = data_batch[: , 0] / (coord_max[0] - coord_min[0])
                normlized_xyz[: , 1] = data_batch[: , 1] / (coord_max[1] - coord_min[1])
                normlized_xyz[: , 2] = data_batch[: , 2] / (coord_max[2] - coord_min[2])

                # move to center
                data_batch[: , 0] = data_batch[: , 0] - (s_x + self.block_size / 2.0)
                data_batch[: , 1] = data_batch[: , 1] - (s_y + self.block_size / 2.0)

                # color (rgb) normalization
                # data_batch[: , 3: 6] /= 255.0
                
                data_batch = np.concatenate((data_batch, normlized_xyz), axis=1)
                label_batch = labels[point_idxs].astype(int)
                batch_weight = self.labelweights[label_batch]

                # vstack: vertical stack; hstack: horizontal stack
                # empty array cannot be stacked with non empty array
                data_bridge = np.vstack([data_bridge, data_batch]) if data_bridge.size else data_batch
                # if data_bridge.size == 0, data_bridge = data_batch
                # else (data_bridge.size != 0), data_bridge = np.vstack([data_bridge, data_batch])
                label_bridge = np.hstack([label_bridge, label_batch]) if label_bridge.size else label_batch
                sample_weight = np.hstack([sample_weight, batch_weight]) if label_bridge.size else batch_weight
                index_bridge = np.hstack([index_bridge, point_idxs]) if index_bridge.size else point_idxs
        data_bridge = data_bridge.reshape((-1, self.block_points, data_bridge.shape[1]))
        label_bridge = label_bridge.reshape((-1, self.block_points))
        sample_weight = sample_weight.reshape((-1, self.block_points))
        index_bridge = index_bridge.reshape((-1, self.block_points))



        return data_bridge, label_bridge, sample_weight, index_bridge

    def __len__(self):
        return len(self.scene_points_list)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import time


    # 测试数据集加载和可视化
    def visualize_block(dataset, block_idx=0):
        print(f"\n正在可视化第 {block_idx} 个数据块...")

        # 获取一个数据块
        start_time = time.time()
        data = dataset[block_idx]
        load_time = time.time() - start_time
        print(f"数据加载时间: {load_time:.2f} 秒")

        # 打印数据块信息
        points = data['points']
        colors = data['colors']/65535
        labels = data['labels']

        print("\n数据块统计信息:")
        print(f"点数: {len(points)}")
        print(f"点云范围:")
        print(f"X: [{points[:, 0].min():.2f}, {points[:, 0].max():.2f}]")
        print(f"Y: [{points[:, 1].min():.2f}, {points[:, 1].max():.2f}]")
        print(f"Z: [{points[:, 2].min():.2f}, {points[:, 2].max():.2f}]")
        print(labels)

        label_counts = np.bincount(labels, minlength=5)

        # 打印每个标签的数量
        for i in range(5):
            print(f"Label {i}: {label_counts[i]}")

        # 创建3D可视化图
        fig = plt.figure(figsize=(15, 5))

        # 1. 使用坐标显示
        ax1 = fig.add_subplot(131, projection='3d')
        scatter1 = ax1.scatter(points[:, 0], points[:, 1], points[:, 2],
                             c=labels,
                             cmap='tab20', s=1)
        ax1.set_title('PCD coordinate view')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')

        # 2. 使用颜色显示
        ax2 = fig.add_subplot(132, projection='3d')
        scatter2 = ax2.scatter(points[:, 0], points[:, 1], points[:, 2],
                             c=colors,  # RGB颜色
                             s=1)
        ax2.set_title('PCD color view')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')

        # 3. 使用标签显示
        ax3 = fig.add_subplot(133, projection='3d')
        scatter3 = ax3.scatter(points[:, 0], points[:, 1], points[:, 2],
                             c=labels,
                             cmap='tab20', s=1)
        ax3.set_title('PCD label view')
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')
        ax3.set_zlabel('Z')

        # 添加颜色条
        plt.colorbar(scatter1, ax=ax1, label='Labels')
        plt.colorbar(scatter3, ax=ax3, label='Labels')

        plt.tight_layout()
        plt.show()

        # 打印数据形状
        print("\n数据形状:")
        print(f"Points shape: {points.shape}")
        print(f"Colors shape: {colors.shape}")
        print(f"Labels shape: {labels.shape}")

        return data


    specific_file = ['bridge-7.las']
    # 创建数据集实例

    root = '../data/fukushima/onepart'
    dataset = LWBridgeDataset(split='val', data_root=root, num_point=4096, block_size=1.0,
                                    sample_rate=1.0, num_class=4096, transform=None)

    # data = dataset[200]
    # labels = data['labels']
    # # 统计每个标签的数量
    # label_counts = np.bincount(labels, minlength=5)
    #
    # # 打印每个标签的数量
    # for i in range(5):
    #     print(f"Label {i}: {label_counts[i]}")

    # 可视化第一个数据块
    block_data = visualize_block(dataset, block_idx=4900).time()