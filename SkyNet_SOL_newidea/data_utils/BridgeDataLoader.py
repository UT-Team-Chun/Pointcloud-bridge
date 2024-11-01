import os
import numpy as np
from torch.utils.data import Dataset
import laspy

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

        assert split in ['train', 'test']
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
            
        return current_points, current_labels

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
    up_up_root = os.path.dirname(os.path.dirname(__file__))
    data_root = 'data/by_iPhone/'
    root_total = os.path.join(up_up_root, data_root)
    num_point, block_size, sample_rate = 4096, 1.0, 0.01

    point_data = LWBridgeDataset(split='train', data_root=root_total, num_point=num_point, block_size=block_size, sample_rate=sample_rate, transform=None)
    print('point data size:', point_data.__len__())
    print('point data 0 shape:', point_data.__getitem__(0)[0].shape)
    print('point label 0 shape:', point_data.__getitem__(0)[1].shape)
    import torch, time, random
    manual_seed = 123
    random.seed(manual_seed)
    np.random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    torch.cuda.manual_seed_all(manual_seed)
    def worker_init_fn(worker_id):
        random.seed(manual_seed + worker_id)
    # train_loader = torch.utils.data.DataLoader(point_data, batch_size=16, shuffle=True, num_workers=16, pin_memory=True, worker_init_fn=worker_init_fn)
    train_loader = torch.utils.data.DataLoader(point_data, batch_size=16, shuffle=True, num_workers=16, pin_memory=True)
    for idx in range(4):
        end = time.time()
        for i, (input, target) in enumerate(train_loader):
            print('time: {}/{}--{}'.format(i + 1, len(train_loader), time.time() - end))
            end = time.time()