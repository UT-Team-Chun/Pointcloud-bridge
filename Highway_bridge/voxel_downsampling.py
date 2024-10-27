import numpy as np
import laspy
import open3d as o3d
from tqdm import tqdm
import time
from pathlib import Path
import sys
import os

def get_file_size(file_path):
    """获取文件大小并转换为合适的单位"""
    size_bytes = os.path.getsize(file_path)
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"

def voxel_downsample_optimized(points, colors, labels, voxel_size=0.008):
    """优化后的体素下采样函数"""
    # 创建点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # 进行下采样
    downsampled_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    
    # 获取下采样后的点和颜色
    downsampled_points = np.asarray(downsampled_pcd.points)
    downsampled_colors = np.asarray(downsampled_pcd.colors)
    
    # 使用KDTree进行最近邻搜索
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    
    # 批量处理标签
    downsampled_labels = np.zeros(len(downsampled_points), dtype=labels.dtype)
    batch_size = 10000
    
    for i in range(0, len(downsampled_points), batch_size):
        batch_end = min(i + batch_size, len(downsampled_points))
        batch_points = downsampled_points[i:batch_end]
        
        for j, point in enumerate(batch_points):
            _, idx, _ = pcd_tree.search_knn_vector_3d(point, 1)
            downsampled_labels[i + j] = labels[idx[0]]
    
    return downsampled_points, downsampled_colors, downsampled_labels

def process_las_file(input_path, output_path, voxel_size=0.008):
    """处理单个LAS文件"""
    try:
        # 读取LAS文件
        las = laspy.read(input_path)
        
        # 提取点云数据
        points = np.vstack((las.x, las.y, las.z)).transpose()
        colors = np.vstack((las.red, las.green, las.blue)).transpose() / 65535.0
        labels = las.classification
        
        # 下采样
        downsampled_points, downsampled_colors, downsampled_labels = voxel_downsample_optimized(
            points, colors, labels, voxel_size
        )
        
        # 创建新的LAS文件
        new_las = laspy.create(point_format=las.header.point_format, file_version=las.header.version)
        
        # 设置点数据
        new_las.header.offsets = las.header.offsets
        new_las.header.scales = las.header.scales
        
        new_las.x = downsampled_points[:, 0]
        new_las.y = downsampled_points[:, 1]
        new_las.z = downsampled_points[:, 2]
        
        new_las.red = (downsampled_colors[:, 0] * 65535).astype(np.uint16)
        new_las.green = (downsampled_colors[:, 1] * 65535).astype(np.uint16)
        new_las.blue = (downsampled_colors[:, 2] * 65535).astype(np.uint16)
        new_las.classification = downsampled_labels
        
        # 保存文件
        new_las.write(output_path)
        
        return len(points), len(downsampled_points)
    
    except Exception as e:
        print(f"Error processing file {input_path}: {str(e)}")
        import traceback
        traceback.print_exc()
        return 0, 0

def batch_process_files(input_folder, output_folder, voxel_size=0.008):
    """批量处理文件"""
    start_time = time.time()
    
    # 创建输出文件夹
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    # 获取所有las文件
    las_files = list(Path(input_folder).glob("*.las"))
    total_files = len(las_files)
    
    if total_files == 0:
        print("No LAS files found in the input folder!")
        return
    
    print(f"\nStarting batch processing:")
    print(f"Found {total_files} LAS files")
    print(f"Voxel size: {voxel_size}m")
    print(f"Input folder: {input_folder}")
    print(f"Output folder: {output_folder}")
    print("="*50)
    
    # 处理统计
    total_original = 0
    total_downsampled = 0
    successful_files = 0
    failed_files = []
    file_stats = []
    
    # 使用tqdm显示总体进度
    for i, input_file in enumerate(tqdm(las_files, desc="Processing files", unit="file")):
        output_file = Path(output_folder) / input_file.name
        
        # 显示当前处理的文件信息
        print(f"\nProcessing {i+1}/{total_files}: {input_file.name}")
        print(f"Input size: {get_file_size(input_file)}")
        
        file_start_time = time.time()
        
        try:
            original_count, downsampled_count = process_las_file(
                str(input_file), 
                str(output_file), 
                voxel_size
            )
            
            if original_count > 0 and os.path.exists(output_file):
                total_original += original_count
                total_downsampled += downsampled_count
                successful_files += 1
                
                # 计算单个文件的统计信息
                processing_time = time.time() - file_start_time
                compression_ratio = (1 - downsampled_count/original_count) * 100
                output_size = get_file_size(output_file)
                
                file_stats.append({
                    'name': input_file.name,
                    'original_points': original_count,
                    'downsampled_points': downsampled_count,
                    'compression_ratio': compression_ratio,
                    'processing_time': processing_time,
                    'output_size': output_size
                })
                
                print(f"Completed: {input_file.name}")
                print(f"Points: {original_count:,} → {downsampled_count:,}")
                print(f"Compression: {compression_ratio:.1f}%")
                print(f"Output size: {output_size}")
                print(f"Time: {processing_time:.2f}s")
            else:
                failed_files.append(input_file.name)
                
        except Exception as e:
            print(f"Error processing {input_file.name}: {str(e)}")
            failed_files.append(input_file.name)
    
    # 打印详细的处理报告
    total_time = time.time() - start_time
    print("\n" + "="*50)
    print("Processing Report:")
    print("="*50)
    print(f"Total processing time: {total_time:.2f} seconds")
    print(f"Successfully processed: {successful_files}/{total_files} files")
    print(f"Total original points: {total_original:,}")
    print(f"Total downsampled points: {total_downsampled:,}")
    print(f"Overall compression ratio: {(1 - total_downsampled/total_original)*100:.1f}%")
    
    # 打印每个文件的详细信息
    print("\nPer-file statistics:")
    print("-"*50)
    for stat in file_stats:
        print(f"\nFile: {stat['name']}")
        print(f"Points: {stat['original_points']:,} → {stat['downsampled_points']:,}")
        print(f"Compression: {stat['compression_ratio']:.1f}%")
        print(f"Output size: {stat['output_size']}")
        print(f"Processing time: {stat['processing_time']:.2f}s")
    
    if failed_files:
        print("\nFailed files:")
        for file in failed_files:
            print(f"- {file}")
    
    print("="*50)

if __name__ == "__main__":
    input_folder = "./raw-1"
    output_folder = "./downsampled"
    voxel_size = 0.02  # 20mm
    
    # 检查输入文件夹是否存在
    if not os.path.exists(input_folder):
        print(f"Error: Input folder '{input_folder}' does not exist!")
        sys.exit(1)
    
    # 开始批处理
    batch_process_files(input_folder, output_folder, voxel_size)
