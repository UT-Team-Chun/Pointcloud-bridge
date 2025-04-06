import os
import sys
import numpy as np
import laspy
from tqdm import tqdm

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

    
    input_dir = 'data/shuto-E/down_8c_voxel005'
    output_dir = 'data/shuto-E/down_5c'

    process_directory(input_dir, output_dir)
