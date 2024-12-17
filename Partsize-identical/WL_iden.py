import csv
import logging
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.spatial import ConvexHull, cKDTree
from sklearn.decomposition import PCA
from sklearn.linear_model import RANSACRegressor
from tqdm import tqdm

from tool_utils.load_las import read_las_file


def load_data(name, label):
    # 获取当前文件的路径
    current_file = Path(__file__)

    # 获取项目根目录（向上两级）
    root_dir = current_file.parent.parent
    root_project = current_file.parent
    # 构建data目录的路径
    data_dir = root_dir / 'data' / 'bridge-5cls-fukushima' / 'inference_data'

    # 读取数据
    pathraw = data_dir / 'raw' / f"{name}_test.las"
    pathtest = data_dir / 'pred' / f"{name}_pred.las"

    data_raw = read_las_file(pathraw)
    data_test = read_las_file(pathtest)

    # 提取床板数据（label == 2）
    #{'abutment': 0, 'girder': 1, 'deck': 2, 'parapet': 3, 'noise': 4}
    deck_raw = data_raw[data_raw[:, 6] == label]
    deck_test = data_test[data_test[:, 6] == label]

    # 检查是否存在指定的label数据
    if deck_raw.size == 0 or deck_test.size == 0:
        return None, None

    print(f"Total points: {data_test.shape[0]}")
    print(f"Deck points_raw: {deck_raw.shape}")
    print(f"Deck points_test: {deck_test.shape}")
    log_string(f"Rawdata is : {name}_test.las")
    log_string(f"Testdata is : {name}_pred.las")

    return deck_raw, deck_test

def ransac_plane_fit(points, max_trials=2000, residual_threshold=0.1):
    ransac = RANSACRegressor(max_trials=max_trials, residual_threshold=residual_threshold, random_state=42)
    ransac.fit(points[:, :2], points[:, 2])
    inlier_mask = ransac.inlier_mask_
    return points[inlier_mask]

def project_to_plane(points):
    return points[:, :2]

def align_to_principal_axes(points):
    pca = PCA(n_components=2)
    pca.fit(points)
    return pca.transform(points)

def detect_and_trim_edges(points, percentile=20):
    x, y = points[:, 0], points[:, 1]
    x_density, x_bins = np.histogram(x, bins=100)
    y_density, y_bins = np.histogram(y, bins=100)
    
    x_threshold = np.percentile(x_density, percentile)
    y_threshold = np.percentile(y_density, percentile)
    
    x_indices = np.clip(np.digitize(x, x_bins[1:-1]) - 1, 0, len(x_density) - 1)
    y_indices = np.clip(np.digitize(y, y_bins[1:-1]) - 1, 0, len(y_density) - 1)
    
    x_mask = np.logical_and(x_density[x_indices] > x_threshold, x_density[x_indices] < np.max(x_density))
    y_mask = np.logical_and(y_density[y_indices] > y_threshold, y_density[y_indices] < np.max(y_density))
    
    return points[np.logical_and(x_mask, y_mask)]

def minimum_bounding_rectangle(points):
    hull_points = points[ConvexHull(points).vertices]
    edges = np.subtract.outer(hull_points, hull_points).reshape(-1, 2)
    angles = np.arctan2(edges[:, 1], edges[:, 0])
    angles = np.abs(np.mod(angles, np.pi/2))
    angles = np.unique(angles)
    
    rotations = np.vstack([np.cos(angles), -np.sin(angles), 
                           np.sin(angles), np.cos(angles)]).T
    rotations = rotations.reshape((-1, 2, 2))
    
    rot_points = np.dot(rotations, hull_points.T)
    
    min_x = np.nanmin(rot_points[:, 0], axis=1)
    max_x = np.nanmax(rot_points[:, 0], axis=1)
    min_y = np.nanmin(rot_points[:, 1], axis=1)
    max_y = np.nanmax(rot_points[:, 1], axis=1)
    
    areas = (max_x - min_x) * (max_y - min_y)
    best_idx = np.argmin(areas)
    
    x1, x2 = max_x[best_idx], min_x[best_idx]
    y1, y2 = max_y[best_idx], min_y[best_idx]
    r = rotations[best_idx]
    
    rval = np.array([
        np.dot([x1, y2], r),
        np.dot([x2, y2], r),
        np.dot([x2, y1], r),
        np.dot([x1, y1], r)
    ])
    
    return rval


def adaptive_voxel_size(data, target_points_ratio=0.1, min_points=1000, max_voxel_size=0.5, min_voxel_size=0.01):
    """
    自适应计算合适的体素大小

    参数:
    data: 输入点云数据
    target_points_ratio: 目标保留点的比例（默认0.1，即保留10%的点）
    min_points: 最少保留的点数
    max_voxel_size: 最大体素大小
    min_voxel_size: 最小体素大小

    返回:
    float: 建议的体素大小
    """
    points = data[:, :3]

    # 1. 计算点云的基本特征
    n_points = len(points)
    bbox_min = np.min(points, axis=0)
    bbox_max = np.max(points, axis=0)
    diagonal_length = np.linalg.norm(bbox_max - bbox_min)
    point_density = n_points / (np.prod(bbox_max - bbox_min))

    # 2. 计算点云的平均最近邻距离（使用随机采样来加速）
    sample_size = min(1000, n_points)
    sampled_points = points[np.random.choice(n_points, sample_size, replace=False)]
    tree = cKDTree(sampled_points)
    distances, _ = tree.query(sampled_points, k=2)  # k=2因为最近的是点本身
    mean_nn_distance = np.mean(distances[:, 1])  # 使用第二近的点

    # 3. 初始体素大小估计
    # 基于点密度的估计
    density_based_size = (1 / point_density) ** (1 / 3)
    # 基于最近邻距离的估计
    nn_based_size = mean_nn_distance * 2

    # 4. 结合两种估计
    initial_voxel_size = np.mean([density_based_size, nn_based_size])

    # 5. 根据目标点数调整
    target_points = max(min_points, int(n_points * target_points_ratio))

    # 使用二分查找找到合适的体素大小
    voxel_size = initial_voxel_size
    left = min_voxel_size
    right = max_voxel_size

    for _ in range(10):  # 最多尝试10次
        voxel_coords = np.floor(points / voxel_size).astype(int)
        unique_coords = np.unique(voxel_coords, axis=0)
        current_points = len(unique_coords)

        if abs(current_points - target_points) / target_points < 0.1:  # 误差在10%以内就接受
            break

        if current_points > target_points:
            left = voxel_size
            voxel_size = (voxel_size + right) / 2
        else:
            right = voxel_size
            voxel_size = (left + voxel_size) / 2

    # 确保在合理范围内
    voxel_size = np.clip(voxel_size, min_voxel_size, max_voxel_size)

    return voxel_size


def data_voxel(data, voxel_size=None):
    """
    改进的体素化函数，支持自适应体素大小
    """
    if voxel_size is None:
        voxel_size = adaptive_voxel_size(data)
        print(f"Using adaptive voxel size: {voxel_size:.3f}")

    bridge_points = data[:, :3]
    voxel_coords = np.floor(bridge_points / voxel_size).astype(int)
    _, unique_indices = np.unique(voxel_coords, axis=0, return_index=True)
    filtered_points = bridge_points[unique_indices]

    return filtered_points


def isolation_forest_outlier_removal(points, contamination=0.1):
    """
    改进的Isolation Forest异常检测，考虑方向性
    """
    from sklearn.ensemble import IsolationForest

    # 桥长方向检测（更宽松）
    points_transformed_length, length_idx, length_contamination, pca = directional_outlier_detection(
        points, contamination, is_length_direction=True
    )

    # 桥宽方向检测（更严格）
    points_transformed_width, width_idx, width_contamination, _ = directional_outlier_detection(
        points, contamination, is_length_direction=False
    )

    #length_contamination=contamination
    #width_contamination=contamination

    # 分别对两个方向进行异常检测
    iso_forest_length = IsolationForest(contamination=length_contamination, random_state=42)
    iso_forest_width = IsolationForest(contamination=width_contamination, random_state=42)

    # 获取预测结果
    length_labels = iso_forest_length.fit_predict(points_transformed_length[:, [length_idx]])
    width_labels = iso_forest_width.fit_predict(points_transformed_width[:, [width_idx]])

    # 组合两个方向的结果（只有两个方向都认为是正常点才保留）
    combined_mask = (length_labels == 1) & (width_labels == 1)

    return points[combined_mask]


def directional_outlier_detection(points, contamination=0.1, is_length_direction=True):
    """
    根据方向性进行异常点检测

    参数:
    points: 输入点云
    contamination: 异常点比例
    is_length_direction: 是否是桥长方向
    """
    # 使用PCA找到主方向
    pca = PCA(n_components=points.shape[1])
    points_transformed = pca.fit_transform(points)

    # 根据方差比确定主方向
    variance_ratio = pca.explained_variance_ratio_
    main_direction_idx = 0 if variance_ratio[0] > variance_ratio[1] else 1

    # 确定处理方向
    direction_idx = main_direction_idx if is_length_direction else (1 - main_direction_idx)

    # 调整contamination
    adjusted_contamination = contamination * (0.5 if is_length_direction else 1)

    return points_transformed, direction_idx, adjusted_contamination, pca



def adaptive_lof_params(points, target_precision=0.03, min_neighbors=5, max_neighbors=50):
    """
    自适应确定LOF的参数

    参数:
    points: 输入点云数据 shape=(N, 3)
    target_precision: 目标精度（米），默认3cm
    min_neighbors: 最小邻居数
    max_neighbors: 最大邻居数

    返回:
    tuple: (n_neighbors, contamination)
    """
    from sklearn.neighbors import NearestNeighbors
    import numpy as np

    # 1. 计算点云基本特征
    n_points = len(points)
    bbox_min = np.min(points, axis=0)
    bbox_max = np.max(points, axis=0)
    volume = np.prod(bbox_max - bbox_min)
    point_density = n_points / volume

    # 2. 估计局部点密度
    # 在目标精度范围内应该包含的点数
    expected_points = point_density * (4 / 3 * np.pi * (target_precision ** 3))

    # 3. 计算点的分布特征
    # 使用随机采样来加速计算
    sample_size = min(1000, n_points)
    sampled_points = points[np.random.choice(n_points, sample_size, replace=False)]

    # 计算k近邻距离
    k = min(20, len(sampled_points) - 1)
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(sampled_points)
    distances, _ = nbrs.kneighbors()

    # 计算距离的统计特征
    mean_dist = np.mean(distances[:, 1:], axis=1)  # 排除自身
    std_dist = np.std(distances[:, 1:], axis=1)

    # 4. 估计最佳邻居数
    # 基于点密度的估计
    density_based_k = int(expected_points)

    # 基于距离分布的估计
    # 使用变异系数（CV）来评估局部分布的均匀性
    cv = std_dist / mean_dist
    mean_cv = np.mean(cv)

    # 如果变异系数大，说明点分布不均匀，需要更多的邻居
    cv_factor = 1 + mean_cv

    # 综合考虑密度和分布特征
    n_neighbors = int(density_based_k * cv_factor)

    # 限制在合理范围内
    n_neighbors = np.clip(n_neighbors, min_neighbors, max_neighbors)

    # 5. 估计contamination
    # 基于距离分布估计异常点比例
    threshold = np.mean(mean_dist) + 2 * np.std(mean_dist)
    outlier_ratio = np.mean(mean_dist > threshold)

    # 限制contamination的范围
    contamination = np.clip(outlier_ratio, 0.01, 0.1)

    return n_neighbors, contamination


def lof_outlier_removal(points, n_neighbors=None, contamination=None):
    """
    改进的LOF异常点去除函数，支持自适应参数
    """
    from sklearn.neighbors import LocalOutlierFactor

    if n_neighbors is None or contamination is None:
        n_neighbors, contamination = adaptive_lof_params(points)
        print(f"Using adaptive LOF parameters: n_neighbors={n_neighbors}, contamination={contamination:.3f}")

    lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
    outlier_labels = lof.fit_predict(points)

    inliers = points[outlier_labels == 1]

    return inliers


def dbscan_outlier_removal(points, eps=0.5, min_samples=5):
    from sklearn.cluster import DBSCAN
    from sklearn.preprocessing import StandardScaler
    # 标准化数据
    scaler = StandardScaler()
    points_scaled = scaler.fit_transform(points)

    # 应用DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = dbscan.fit_predict(points_scaled)

    # 识别非离群点（即被分配到簇中的点）
    inliers = points[clusters != -1] # -1 表示离群点
    outliers = points[clusters == -1]

    return inliers

def process_bridge_deck(points, voxel_size=0.02, ransac_max_trials=1000, ransac_residual_threshold=0.3,
                        isolation_forest_contamination=0.3, lof_n_neighbors=30, lof_contamination=0.4,
                        dbscan_eps=1, dbscan_min_samples=5,percentile=20):
    # 只使用 x, y, z 坐标
    result = points[:, :3]

    with tqdm(total=10, desc="Processing pred data", leave=True) as pbar:
        # 下采样
        result = data_voxel(result, voxel_size=voxel_size)
        pbar.update(1)

        # 1. analysis
        result = ransac_plane_fit(result, max_trials=ransac_max_trials, residual_threshold=ransac_residual_threshold)
        pbar.update(1)

        result = isolation_forest_outlier_removal(result, contamination=isolation_forest_contamination)
        pbar.update(1)

        result = lof_outlier_removal(result, n_neighbors=lof_n_neighbors, contamination=lof_contamination)
        pbar.update(1)

        #result = dbscan_outlier_removal(result, eps=dbscan_eps, min_samples=dbscan_min_samples)
        pbar.update(1)

        result = project_to_plane(result)
        pbar.update(1)

        # # 应用三种算法
        # # Isolation Forest
        # iforest = IsolationForest(contamination=0.1, random_state=42)
        # iforest_pred = iforest.fit_predict(X)
        #
        # # LOF
        # lof = LocalOutlierFactor(contamination=0.1)
        # lof_pred = lof.fit_predict(X)
        #
        # # DBSCAN
        # dbscan = DBSCAN(eps=0.3, min_samples=5)
        # dbscan_pred = dbscan.fit_predict(X)
        # # 将DBSCAN的结果转换为异常检测格式（-1为异常，其他为正常）
        # dbscan_outliers = np.where(dbscan_pred == -1, -1, 1)
        #
        # # 组合结果（投票机制）
        # # 如果至少两个算法认为是异常点，则标记为异常
        # combined_pred = np.where((iforest_pred + lof_pred + dbscan_outliers) < -1, -1, 1)

        # 2. 主方向对齐
        #result = align_to_principal_axes(result)
        #pbar.update(1)

        # 3. 边缘检测和修剪
        points_trimmed = detect_and_trim_edges(result,percentile)
        result = detect_and_trim_edges(result)
        pbar.update(1)

        # 4. 矩形拟合
        rect = minimum_bounding_rectangle(result)
        pbar.update(1)

        # 计算长度和宽度
        width = np.linalg.norm(rect[1] - rect[0])
        length = np.linalg.norm(rect[2] - rect[1])
        pbar.update(1)

        length, width = calculate_dimensions(result, rect)
    return max(width, length), min(width, length), points_trimmed, rect



def process_raw(points,percentile=20):
    # 只使用 x, y, z 坐标
    xyz_points = points[:, :3]

    # 使用 tqdm 显示进度
    with tqdm(total=5, desc="Processing raw data", leave=True) as pbar:
        result = project_to_plane(xyz_points)
        pbar.update(1)

        # 2. 主方向对齐
        result = align_to_principal_axes(result)
        pbar.update(1)

        # 3. 边缘检测和修剪
        points_trimmed = detect_and_trim_edges(result,percentile)
        result = detect_and_trim_edges(result)
        pbar.update(1)

        # 4. 矩形拟合
        rect = minimum_bounding_rectangle(result)
        pbar.update(1)

        # 计算长度和宽度
        width = np.linalg.norm(rect[1] - rect[0])
        length = np.linalg.norm(rect[2] - rect[1])
        pbar.update(1)

        length, width = calculate_dimensions(result, rect)
    return max(width, length), min(width, length), points_trimmed, rect
    #return length, width, points_trimmed, rect


def evaluate_result(length_raw, width_raw, length_processed, width_processed):
    length_error = abs(length_raw - length_processed) / length_raw
    width_error = abs(width_raw - width_processed) / width_raw
    return (length_error + width_error) / 2  # 平均相对误差


# 统计学评价指标
def statistical_evaluation(df):
    stats = df.describe()
    correlation = df.corr()
    return stats, correlation

# 可视化评价结果
def visualize_evaluation(df):
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix')
    plt.savefig('./fig/correlation_matrix.png', dpi=400, bbox_inches='tight', pad_inches=0.1)
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.pairplot(df)
    plt.savefig('./fig/pairplot.png', dpi=400, bbox_inches='tight', pad_inches=0.1)
    plt.show()


def calculate_dimensions(points, rect):
    """
    通过最小外接矩形的边缘点计算长宽，并进行微小的修正

    参数:
    points: np.array, 点云数据 (N, 2)
    rect: np.array, 最小外接矩形的四个顶点 (4, 2)

    返回:
    tuple: (length, width) 最终的长度和宽度
    """
    # 1. 首先获取原始的长宽（这个是比较准的基准）
    original_width = np.linalg.norm(rect[1] - rect[0])
    original_length = np.linalg.norm(rect[2] - rect[1])

    # 2. 计算矩形的主方向
    edge1 = rect[1] - rect[0]  # 宽度方向
    edge2 = rect[2] - rect[1]  # 长度方向

    # 归一化方向向量
    dir1 = edge1 / np.linalg.norm(edge1)
    dir2 = edge2 / np.linalg.norm(edge2)

    # 3. 将靠近边缘的点投影（只考虑接近边缘的点）
    margin = 0.1  # 边缘区域的范围（占总长度的比例）

    # 找到靠近各边的点
    proj1 = np.dot(points - rect[0], dir1)
    proj2 = np.dot(points - rect[1], dir2)

    # 只选择靠近边缘的点
    edge_points_width = points[
        (proj1 < margin * original_width) |
        (proj1 > (1 - margin) * original_width)
        ]

    edge_points_length = points[
        (proj2 < margin * original_length) |
        (proj2 > (1 - margin) * original_length)
        ]

    # 4. 计算边缘点的投影
    if len(edge_points_width) > 0 and len(edge_points_length) > 0:
        # 计算边缘点投影的范围
        width_proj = np.dot(edge_points_width - rect[0], dir1)
        length_proj = np.dot(edge_points_length - rect[1], dir2)

        # 使用边缘点的范围，但限制修正幅度
        width = np.clip(
            np.max(width_proj) - np.min(width_proj),
            0.95 * original_width,
            1.05 * original_width
        )
        length = np.clip(
            np.max(length_proj) - np.min(length_proj),
            0.95 * original_length,
            1.05 * original_length
        )
    else:
        # 如果没有足够的边缘点，使用原始值
        width = original_width
        length = original_length

    return length, width


# # 主程序
if __name__ == "__main__":
    def log_string(str):
        logger.info(str)
        print(str)

    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('./log/identical/log.txt', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')

    test_names = ['cb2-5c', 'cb6-5c','cb9-5c'] #b1,b2,b7
    #test_names = ['b1','b2','b7'] #b1,b2,b7 ,'cb2-5c', 'cb6-5c','cb9-5c'
    #{'abutment': 0, 'girder': 1, 'deck': 2, 'parapet': 3, 'noise': 4}
    #{'abutment': 1, 'girder': 2, 'deck': 3, 'parapet': 4, 'noise': 0}
    label = [2,3,4]
    #label = [1,2,3]
    total_error = 0
    total_time = 0

    # 超参数
    voxel_size = 0.05 #best 0.3 for cb is 0.05, for b is 0.3
    ransac_max_trials = 1000 #best 1000
    ransac_residual_threshold = 0.3 #best 0.3
    isolation_forest_contamination = 0.03 #best 0.05
    lof_n_neighbors = 30 #10 #best 30 for cb is 30, for b is 10
    lof_contamination = 'auto' #'auto',best 0.4
    dbscan_eps = 1  #best 1
    dbscan_min_samples = 5 #best 5
    percentile = 25
    note='no_PCA,CB'
    # 创建CSV文件并写入表头
    with open('evaluation_results.csv', mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow([])  # 空行
        writer.writerow([])  # 空行
        writer.writerow(
            ['Voxel Size', 'RANSAC Max Trials', 'RANSAC Residual Threshold', 'Isolation Forest Contamination',
             'LOF Neighbors', 'LOF Contamination', 'DBSCAN EPS', 'DBSCAN Min Samples','percentile','note'])
        writer.writerow([voxel_size, ransac_max_trials, ransac_residual_threshold, isolation_forest_contamination,
                         lof_n_neighbors, lof_contamination, dbscan_eps, dbscan_min_samples,percentile,note])
        writer.writerow(['Case', 'Label', 'Original Length', 'Original Width', 'Estimated Length', 'Estimated Width',
                         'Relative Error'])

        for l in label:
            for name in tqdm(test_names):
                deck_raw, deck_test = load_data(name,l)
                if deck_raw is None or deck_test is None:
                    log_string(f"Case {name}-{l}: No data for label {l}, skipping...")
                    continue
                start_time = time.time()

                length_raw, width_raw, cleaned_points, bounding_rect = process_raw(deck_raw,percentile)
                length, width, cleaned_points_test, bounding_rect_test = process_bridge_deck(deck_test, voxel_size,
                                                                                             ransac_max_trials,
                                                                                             ransac_residual_threshold,
                                                                                             isolation_forest_contamination,
                                                                                             lof_n_neighbors,
                                                                                             lof_contamination,
                                                                                             dbscan_eps,
                                                                                             dbscan_min_samples)
                error = evaluate_result(length_raw, width_raw, length, width)

                log_string(f"Case {name}-{l}: 原始的长度: {length_raw:.2f}")
                log_string(f"Case {name}-{l}: 原始的宽度: {width_raw:.2f}")
                log_string(f"Case {name}-{l}: 估计的长度: {length:.2f}")
                log_string(f"Case {name}-{l}: 估计的宽度: {width:.2f}")
                log_string(f"Case {name}-{l}: 相对误差: {error:.2f}")

                total_error += error

                # 写入CSV文件
                writer.writerow([f"{name}-{l}", l, length_raw, width_raw, length, width, error])

                end_time = time.time()
                use_time = end_time - start_time

                log_string(f"Case {name}-{l}: 用时: {use_time:.2f} 秒")

                total_time += use_time

                # 绘制结果
                plt.rcParams.update({
                    'font.size': 14,          # 基础字体大小
                    'axes.titlesize': 16,     # 标题字体大小
                    'axes.labelsize': 14,     # 轴标签字体大小
                    'xtick.labelsize': 12,    # x轴刻度标签大小
                    'ytick.labelsize': 12     # y轴刻度标签大小
                })

                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))  # 约144mm宽

                ax1.scatter(cleaned_points_test[:, 0], cleaned_points_test[:, 1], alpha=0.1, s=1)
                ax1.plot(bounding_rect_test[:, 0], bounding_rect_test[:, 1], 'r-')
                ax1.set_title('Test Data: Bridge Deck Point Cloud')
                ax1.set_xlabel('X')
                ax1.set_ylabel('Y')
                ax1.axis('equal')
                ax1.grid(True)

                ax2.scatter(cleaned_points[:, 0], cleaned_points[:, 1], alpha=0.1, s=1)
                ax2.plot(bounding_rect[:, 0], bounding_rect[:, 1], 'r-')
                ax2.set_title('Original Data: Bridge Deck Point Cloud')
                ax2.set_xlabel('X')
                ax2.set_ylabel('Y')
                ax2.axis('equal')
                ax2.grid(True)

                plt.tight_layout()

                plt.savefig(f'./fig/result_{name}-{l}.png', dpi=400, bbox_inches='tight', pad_inches=0.1)

        MAError = total_error / (len(test_names) * len(label))
        log_string(f"Mean Average Error: {MAError:.2f}")
        writer.writerow(['Mean Average Error', MAError, 'Mean Time', total_time / (len(test_names) * len(label))])