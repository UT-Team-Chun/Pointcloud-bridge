import os
import numpy as np
import laspy
from glob import glob
from tqdm import tqdm

def translate_labels(input_folder, output_folder=None):
    """
    将指定文件夹中所有las文件的标签按照规则替换：
    0->1, 1->2, 2->3, 3->4, 4->0
    
    参数:
    input_folder: 输入文件夹路径，包含las文件
    output_folder: 输出文件夹路径，如果不指定则覆盖原文件
    """
    # 如果输出文件夹不存在且指定了输出路径，则创建
    if output_folder and not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 获取所有las文件
    las_files = glob(os.path.join(input_folder, "*.las"))
    
    if not las_files:
        print(f"在{input_folder}中未找到las文件")
        return
    
    print(f"找到{len(las_files)}个las文件")
    
    # 标签替换映射
    label_map = {0: 1, 1: 2, 2: 3, 3: 4, 4: 0}
    
    # 处理每个文件
    for las_file in tqdm(las_files, desc="处理las文件"):
        try:
            # 读取las文件
            las = laspy.read(las_file)
            
            # 获取当前标签
            current_labels = las.classification
            
            # 替换标签
            new_labels = np.copy(current_labels)
            for old_label, new_label in label_map.items():
                new_labels[current_labels == old_label] = new_label
            
            # 更新标签
            las.classification = new_labels
            
            # 保存文件
            if output_folder:
                output_path = os.path.join(output_folder, os.path.basename(las_file))
            else:
                output_path = las_file
            
            las.write(output_path)
        except Exception as e:
            print(f"处理文件 {las_file} 时出错: {str(e)}")

if __name__ == "__main__":
    input_folder = 'data/fukushima/org/train'
    output_folder = None
    
    if not output_folder:
        output_folder = None
    
    translate_labels(input_folder, output_folder)
