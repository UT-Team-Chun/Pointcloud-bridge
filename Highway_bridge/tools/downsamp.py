import os
import numpy as np
import laspy
import open3d as o3d
from glob import glob
from tqdm import tqdm
import os
import sys
import numpy as np
import laspy
from tqdm import tqdm

def voxel_downsample_las(input_folder, output_folder=None, voxel_size=0.02, analyze_only=False):
    """
    对指定文件夹中的las点云数据进行体素网格降采样
    
    参数:
    input_folder: 输入文件夹路径，包含las文件
    output_folder: 输出文件夹路径，如不指定则默认为input_folder + "_downsampled"
    voxel_size: 体素大小，单位为米。默认0.02m
    analyze_only: 如果为True，只分析不保存文件
    """
    # 如果没有指定输出文件夹，则创建默认输出文件夹
    if output_folder is None and not analyze_only:
        output_folder = input_folder + "_d"
        
    # 创建输出文件夹（如果不存在）
    if not analyze_only and not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 获取所有las文件
    las_files = glob(os.path.join(input_folder, "*.las"))
    
    if not las_files:
        print(f"在{input_folder}中未找到las文件")
        return
    
    print(f"找到{len(las_files)}个las文件")
    
    # 用于统计
    total_points_before = 0
    total_points_after = 0
    
    # 处理每个文件
    for las_file in tqdm(las_files, desc="处理las文件"):
        try:
            # 读取las文件
            las_in = laspy.read(las_file)
            
            # 获取点坐标和标签
            points = np.vstack((las_in.x, las_in.y, las_in.z)).transpose()
            classifications = las_in.classification
            
            # 创建Open3D点云对象
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            
            # 记录原始点数
            original_points = len(pcd.points)
            total_points_before += original_points
            
            # 体素降采样
            downsampled_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
            
            # 记录降采样后点数
            downsampled_points = len(downsampled_pcd.points)
            total_points_after += downsampled_points
            
            # 输出降采样结果
            reduction_ratio = (1 - downsampled_points / original_points) * 100
            print(f"{os.path.basename(las_file)}: {original_points} -> {downsampled_points} 点 "
                  f"(减少了 {reduction_ratio:.2f}%)")
            
            if not analyze_only:
                # 获取降采样后的点坐标
                downsampled_points = np.asarray(downsampled_pcd.points)
                
                # 为降采样后的点分配标签(使用最近邻)
                from scipy.spatial import cKDTree
                tree = cKDTree(points)
                _, indices = tree.query(downsampled_points, k=1)
                downsampled_classifications = classifications[indices]
                
                # 创建新的las文件 - 修改这部分以解决header.copy()的问题
                las_out = laspy.create(point_format=las_in.header.point_format, file_version=las_in.header.version)
                
                # 设置点坐标和标签
                las_out.x = downsampled_points[:, 0]
                las_out.y = downsampled_points[:, 1]
                las_out.z = downsampled_points[:, 2]
                las_out.classification = downsampled_classifications
                
                # 复制其他通道(如果存在)
                if hasattr(las_in, 'intensity') and len(las_in.intensity) > 0:
                    las_out.intensity = las_in.intensity[indices]
                
                # 复制RGB信息(如果存在)
                if hasattr(las_in, 'red') and len(las_in.red) > 0:
                    las_out.red = las_in.red[indices]
                if hasattr(las_in, 'green') and len(las_in.green) > 0:
                    las_out.green = las_in.green[indices]
                if hasattr(las_in, 'blue') and len(las_in.blue) > 0:
                    las_out.blue = las_in.blue[indices]
                
                # 如果需要其他属性，可以类似地添加
                # 例如复制return_number, number_of_returns等
                for attribute in las_in.point_format.dimension_names:
                    if attribute not in ['X', 'Y', 'Z', 'classification', 'intensity', 'red', 'green', 'blue']:
                        try:
                            if hasattr(las_in, attribute.lower()) and len(getattr(las_in, attribute.lower())) > 0:
                                setattr(las_out, attribute.lower(), getattr(las_in, attribute.lower())[indices])
                        except:
                            pass
                
                # 保存文件
                output_path = os.path.join(output_folder, os.path.basename(las_file))
                las_out.write(output_path)
                
        except Exception as e:
            print(f"处理文件 {las_file} 时出错: {str(e)}")
            # 打印更详细的错误信息以便调试
            import traceback
            traceback.print_exc()
    
    # 输出总体降采样结果
    if total_points_before > 0:
        overall_reduction = (1 - total_points_after / total_points_before) * 100
        print(f"\n总体降采样结果: {total_points_before} -> {total_points_after} 点 "
              f"(减少了 {overall_reduction:.2f}%)")
    
def analyze_point_density(input_folder):
    """
    分析文件夹中las文件的点云密度
    """
    las_files = glob(os.path.join(input_folder, "*.las"))
    
    if not las_files:
        print(f"在{input_folder}中未找到las文件")
        return
    
    densities = []
    
    for las_file in tqdm(las_files, desc="分析点云密度"):
        try:
            # 读取las文件
            las = laspy.read(las_file)
            
            # 获取点坐标
            points = np.vstack((las.x, las.y, las.z)).transpose()
            
            # 计算边界框体积
            min_bound = np.min(points, axis=0)
            max_bound = np.max(points, axis=0)
            volume = np.prod(max_bound - min_bound)
            
            # 计算点密度 (点数/立方米)
            if volume > 0:
                density = len(points) / volume
                densities.append(density)
                print(f"{os.path.basename(las_file)}: 点密度 = {density:.2f} 点/立方米")
            
        except Exception as e:
            print(f"分析文件 {las_file} 时出错: {str(e)}")
    
    if densities:
        avg_density = np.mean(densities)
        min_density = np.min(densities)
        max_density = np.max(densities)
        std_density = np.std(densities)
        
        print(f"\n密度统计:")
        print(f"  平均密度: {avg_density:.2f} 点/立方米")
        print(f"  最小密度: {min_density:.2f} 点/立方米")
        print(f"  最大密度: {max_density:.2f} 点/立方米")
        print(f"  密度标准差: {std_density:.2f}")
        
        # 基于平均密度推荐体素大小
        recommended_voxel = 1.0 / np.cbrt(avg_density / 5)  # 目标降低到约1/5密度
        print(f"\n推荐体素大小: {recommended_voxel:.3f} 米")
        print(f"  - 精细降采样 (保留更多细节): {recommended_voxel/2:.3f} 米")
        print(f"  - 中等降采样: {recommended_voxel:.3f} 米")
        print(f"  - 粗糙降采样 (更高效率): {recommended_voxel*2:.3f} 米")

def convert_labels(input_file, output_file):
    """
    将LAS点云文件中的标签按照以下规则转换：
    - 标签7 -> 0
    - 标签5 -> 4
    - 标签6 -> 4
    - 其他标签保持不变
    """
    try:
        # 打开LAS文件
        las = laspy.read(input_file)
        
        # 获取原始分类标签
        classification = las.classification
        
        # 创建一个副本以保存修改后的分类标签
        new_classification = np.copy(classification)
        
        # 应用转换规则
        new_classification[classification == 7] = 0
        new_classification[classification == 5] = 4
        new_classification[classification == 6] = 4
        
        # 更新LAS文件的分类标签
        las.classification = new_classification
        
        # 保存修改后的文件
        las.write(output_file)
        return True
    except Exception as e:
        print(f"处理文件 {input_file} 时出错: {str(e)}")
        return False

def process_directory(input_dir, output_dir):
    """处理整个目录中的LAS文件"""
    # 确保输出文件夹存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有LAS文件
    las_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith('.las') or file.lower().endswith('.laz'):
                input_path = os.path.join(root, file)
                # 计算相对路径以保持目录结构
                rel_path = os.path.relpath(input_path, input_dir)
                output_path = os.path.join(output_dir, rel_path)
                # 确保输出目录存在
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                las_files.append((input_path, output_path))
    
    print(f"找到 {len(las_files)} 个LAS/LAZ文件")
    
    # 处理所有文件，显示进度条
    success_count = 0
    for input_file, output_file in tqdm(las_files, desc="处理进度"):
        if convert_labels(input_file, output_file):
            success_count += 1
    
    print(f"处理完成，成功处理了 {success_count}/{len(las_files)} 个文件")
    print(f"标签转换规则: 7->0, 5->4, 6->4")

if __name__ == "__main__":
    
    
    input_folder = 'data/shuto-E/raw_8c'
    output_folder = 'data/shuto-E/down_8c_voxel005'
    voxel_size = 0.02

    # 分析点云密度
    print("\n===== 点云密度分析 =====")
    analyze_point_density(input_folder)
    
    # 执行降采样
    voxel_downsample_las(input_folder, output_folder, voxel_size)

    # 执行标签转换
    print("\n===== 标签转换 =====")
    input_dir = output_folder
    output_dir = 'data/shuto-E/down_5c'

    process_directory(input_dir, output_dir)
