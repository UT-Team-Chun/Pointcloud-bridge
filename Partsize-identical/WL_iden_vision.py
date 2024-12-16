from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull
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

# def detect_and_trim_edges(points, percentile=5, threshold=0.1):
#     x, y = points[:, 0], points[:, 1]
#     x_density, x_bins = np.histogram(x, bins=100)
#     y_density, y_bins = np.histogram(y, bins=100)
#
#     x_threshold = np.percentile(x_density, percentile)
#     y_threshold = np.percentile(y_density, percentile)
#
#     x_indices = np.clip(np.digitize(x, x_bins[1:-1]) - 1, 0, len(x_density) - 1)
#     y_indices = np.clip(np.digitize(y, y_bins[1:-1]) - 1, 0, len(y_density) - 1)
#
#     x_mask = np.logical_and(x_density[x_indices] > x_threshold, x_density[x_indices] < np.max(x_density))
#     y_mask = np.logical_and(y_density[y_indices] > y_threshold, y_density[y_indices] < np.max(y_density))
#
#     trimmed_points = points[np.logical_and(x_mask, y_mask)]
#
#     # 使用线性回归拟合边缘点
#     from sklearn.linear_model import LinearRegression
#     reg = LinearRegression()
#     reg.fit(trimmed_points[:, 0].reshape(-1, 1), trimmed_points[:, 1])
#     y_pred = reg.predict(trimmed_points[:, 0].reshape(-1, 1))
#
#     # 去除偏离拟合线的点
#     mask = np.abs(trimmed_points[:, 1] - y_pred) < threshold
#     return trimmed_points[mask]

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

def data_voxel(data, voxel_size=0.1):

    bridge_points = data[:, :3]
    
    # 体素网格滤波
    voxel_coords = np.floor(bridge_points / voxel_size).astype(int)
    _, unique_indices = np.unique(voxel_coords, axis=0, return_index=True)
    filtered_points = bridge_points[unique_indices]
    
    return filtered_points

def isolation_forest_outlier_removal(points, contamination=0.1):
    from sklearn.ensemble import IsolationForest
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    outlier_labels = iso_forest.fit_predict(points)
    
    inliers = points[outlier_labels == 1]
    outliers = points[outlier_labels == -1]
    
    return inliers

def lof_outlier_removal(points, n_neighbors=20, contamination='auto'):
    from sklearn.neighbors import LocalOutlierFactor
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
    inliers = points[clusters != -1]
    outliers = points[clusters == -1] # -1 表示离群点

    return inliers


def visualize_step(points, step_name, rect=None, save_path=None, fig_size=(16, 4), dpi=300):
    """
    可视化点云处理的每个步骤，包括3D视图和俯视图，优化用于学术论文展示

    Args:
        points: numpy array, 点云数据
        step_name: str, 处理步骤名称
        rect: numpy array, 最小外接矩形的顶点（如果有）
        save_path: str, 保存路径
        fig_size: tuple, 图像大小
        dpi: int, 图像分辨率
    """
    # 设置字体
    from mplfonts import use_font
    use_font('Noto Serif CJK SC')

    import matplotlib.gridspec as gridspec

    # 设置全局样式
    plt.rcParams.update({
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '--',
        'axes.labelsize': 16,
        'axes.titlesize': 18,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'axes.spines.top': True,
        'axes.spines.right': True,
        'axes.spines.left': True,
        'axes.spines.bottom': True,
        'figure.autolayout': False,  # 关闭自动布局
        'font.family': ['DejaVu Sans', 'Arial', 'Microsoft YaHei', 'SimHei'],
    })

    # 创建图形和网格规范
    fig = plt.figure(figsize=fig_size)
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])
    gs.update(left=0.08, right=0.92, bottom=0.2, top=0.85, wspace=0.2)

    # 3D视图
    ax1 = fig.add_subplot(gs[0], projection='3d')

    # 设置统一的显示范围
    #ax1.set_xlim(-6, 6)
    #ax1.set_ylim(-10, 2)
    #ax1.set_zlim(0, 10)

    # 绘制3D点云
    scatter = ax1.scatter(points[:, 0], points[:, 1],
                          points[:, 2] if points.shape[1] > 2 else np.zeros_like(points[:, 0]),
                          c=points[:, 2] if points.shape[1] > 2 else 'royalblue',
                          cmap='viridis', s=2, alpha=0.7)

    # 添加颜色条并设置样式
    if points.shape[1] > 2:
        cbar = plt.colorbar(scatter, ax=ax1, label='Height (m)')
        cbar.ax.tick_params(labelsize=12)
        cbar.set_label('Height (m)', size=14)

    # 设置轴标签和字体大小
    ax1.set_xlabel('X (m)', fontsize=14, labelpad=10)
    ax1.set_ylabel('Y (m)', fontsize=14, labelpad=10)
    ax1.set_zlabel('Z (m)', fontsize=14, labelpad=10)
    ax1.set_title('3D Point Cloud', pad=20, fontsize=16)

    # 设置刻度字体大小
    ax1.tick_params(axis='both', which='major', labelsize=12)

    # 添加网格
    ax1.grid(True, linestyle='--', alpha=0.3)

    # 设置最佳视角
    ax1.view_init(elev=30, azim=45)

    # 俯视图
    ax2 = fig.add_subplot(gs[1])


    # 绘制2D点云
    ax2.scatter(points[:, 0], points[:, 1], s=2, alpha=0.7,
                c='royalblue', label='Point Cloud')

    # 如果有最小外接矩形，则绘制
    if rect is not None:
        rect = np.vstack((rect, rect[0]))  # 闭合矩形
        ax2.plot(rect[:, 0], rect[:, 1], 'r-', linewidth=2.5,
                 label='Bounding Rectangle')
        ax2.legend(fontsize=12, loc='lower right',
                   frameon=True, fancybox=True, framealpha=0.8)

    # 设置显示范围
    #ax2.set_xlim(-6, 6)
    #ax2.set_ylim(-6, 4)


    # 设置轴标签和字体大小
    ax2.set_xlabel('X (m)', fontsize=14)
    ax2.set_ylabel('Y (m)', fontsize=14)
    ax2.set_title('Top View', pad=20, fontsize=16)

    # 设置刻度字体大小
    ax2.tick_params(axis='both', which='major', labelsize=12)

    # 添加网格
    ax2.grid(True, linestyle='--', alpha=0.3)
    ax2.set_aspect('equal')
    ax2.set_aspect(0.7)
    # 添加总标题
    #plt.suptitle(step_name, y=0.98, fontsize=18, fontweight='bold')

    # 保存图像
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight',
                    format='tiff', facecolor='white')

    plt.close()


def process_bridge_deck(points, output_dir, voxel_size=0.02, ransac_max_trials=1000,
                        ransac_residual_threshold=0.3, isolation_forest_contamination=0.3,
                        lof_n_neighbors=30, lof_contamination=0.4,
                        dbscan_eps=1, dbscan_min_samples=5):
    # 创建输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 只使用 x, y, z 坐标
    result = points[:, :3]

    # 可视化原始数据
    visualize_step(result, "Original Point Cloud",
                   save_path=output_dir / "1_original.tiff")

    with tqdm(total=10, desc="Processing pred data", leave=True) as pbar:
        # 下采样
        result = data_voxel(result, voxel_size=voxel_size)
        visualize_step(result, "After Voxel Downsampling",
                       save_path=output_dir / "2_voxel.tiff")
        pbar.update(1)


        # RANSAC平面拟合
        result_ransac = ransac_plane_fit(result, max_trials=ransac_max_trials,
                                         residual_threshold=ransac_residual_threshold)
        visualize_step(result_ransac, "After RANSAC Plane Fitting",
                       save_path=output_dir / "3_ransac.tiff")
        pbar.update(1)

        # Isolation Forest
        result_iso = isolation_forest_outlier_removal(result_ransac,
                                                      contamination=isolation_forest_contamination)
        visualize_step(result_iso, "After Isolation Forest",
                       save_path=output_dir / "4_isolation_forest.tiff")
        pbar.update(1)

        # LOF
        result_lof = lof_outlier_removal(result_iso, n_neighbors=lof_n_neighbors,
                                         contamination=lof_contamination)
        visualize_step(result_lof, "After LOF",
                       save_path=output_dir / "5_lof.tiff")
        pbar.update(1)

        # DBSCAN
        #result_dbscan = dbscan_outlier_removal(result_lof, eps=dbscan_eps,
        #                                       min_samples=dbscan_min_samples)
        #visualize_step(result_dbscan, "After DBSCAN",
        #               save_path=output_dir / "6_dbscan.tiff")
        #pbar.update(1)


        # 投影到平面
        result_proj = project_to_plane(result_lof)
        # 为了保持3D可视化效果，我们保留原始的z坐标
        result_proj_3d = np.column_stack((result_proj, result_lof[:, 2]))
        # visualize_step(result_proj_3d, "After Projection to Plane",
        #                save_path=output_dir / "7_projection.tiff")
        pbar.update(1)

        # # 主方向对齐
        result_pca = align_to_principal_axes(result_proj)
        result_pca = move_point2center(result_pca)
        # result_pca_3d = np.column_stack((result_pca, result_dbscan[:, 2]))
        # visualize_step(result_pca_3d, "After PCA Alignment",
        #                save_path=output_dir / "8_pca.tiff")
        # pbar.update(1)

        # 边缘检测和修剪
        points_trimmed = detect_and_trim_edges(result_pca)
        points_trimmed_3d = np.column_stack((points_trimmed,
                                             np.zeros(len(points_trimmed))))
        visualize_step(points_trimmed_3d, "After Edge Detection and Trimming",
                       save_path=output_dir / "6_edge_trimming.tiff")
        pbar.update(1)

        # 矩形拟合
        rect = minimum_bounding_rectangle(points_trimmed)
        visualize_step(points_trimmed_3d, "Final Result with Bounding Rectangle",
                       rect=rect, save_path=output_dir / "7_final.tiff")
        pbar.update(2)

        # 计算长度和宽度
        width = np.linalg.norm(rect[1] - rect[0])
        length = np.linalg.norm(rect[2] - rect[1])

    return max(width, length), min(width, length), points_trimmed, rect


def process_raw(points,output_dir):

    # 只使用 x, y, z 坐标
    xyz_points = points[:, :3]

    output_dir = Path(output_dir)
    # 使用 tqdm 显示进度
    with tqdm(total=5, desc="Processing raw data", leave=True) as pbar:
        result = project_to_plane(xyz_points)
        pbar.update(1)

        # 2. 主方向对齐
        result_pca = align_to_principal_axes(result)
        result_pca = move_point2center(result_pca)
        result_pca_3d = np.column_stack((result_pca, xyz_points[:, 2]))
        pbar.update(1)

        #3. 边缘检测和修剪
        points_trimmed = detect_and_trim_edges(result)
        points_trimmed_3d = np.column_stack((points_trimmed,
                                            np.zeros(len(points_trimmed))))

        pbar.update(1)

        # 4. 矩形拟合
        rect = minimum_bounding_rectangle(result_pca)
        visualize_step(result_pca_3d, "raw",rect=rect,
                       save_path=output_dir / "bridge_raw.tiff")
        pbar.update(1)

        # 计算长度和宽度
        width = np.linalg.norm(rect[1] - rect[0])
        length = np.linalg.norm(rect[2] - rect[1])
        pbar.update(1)

    return max(width, length), min(width, length), points_trimmed, rect


def move_point2center(points):
    min_vals = np.min(points, axis=0)
    max_vals = np.max(points, axis=0)
    center = (min_vals + max_vals) / 2
    points = points - center
    return points

def evaluate_result(length_raw, width_raw, length_processed, width_processed):
    length_error = abs(length_raw - length_processed) / length_raw
    width_error = abs(width_raw - width_processed) / width_raw
    return (length_error + width_error) / 2  # 平均相对误差


# 统计学评价指标
def statistical_evaluation(df):
    stats = df.describe()
    correlation = df.corr()
    return stats, correlation



# # 主程序
if __name__ == "__main__":

    test_names = 'cb2-5c'#'b1','b2','b7', cb2-4c
    l = 2
    #{'abutment': 0, 'girder': 1, 'deck': 2, 'parapet': 3, 'noise': 4}
    label = [1,2,3]
    total_error = 0
    total_time = 0

    # 超参数
    voxel_size = 0.05
    ransac_max_trials = 1000 #best 1000
    ransac_residual_threshold = 0.3 #best 0.3
    isolation_forest_contamination = 0.02 #best 0.3
    lof_n_neighbors = 30 #best 30
    lof_contamination = 'auto' #'auto',best 0.4
    dbscan_eps = 1  #best 1
    dbscan_min_samples = 10 #best 5

    deck_raw, deck_test = load_data(test_names, l)

    # 创建输出目录
    output_dir = "visualization_results/"+test_names

    length, width, cleaned_points_test, bounding_rect_test = process_bridge_deck(
        deck_test,
        output_dir,
        voxel_size,
        ransac_max_trials,
        ransac_residual_threshold,
        isolation_forest_contamination,
        lof_n_neighbors,
        lof_contamination,
        dbscan_eps,
        dbscan_min_samples
    )
    length_raw, width_raw, cleaned_points, bounding_rect = process_raw(deck_raw,output_dir)