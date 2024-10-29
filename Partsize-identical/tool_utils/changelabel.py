import laspy
import numpy as np
from tqdm import tqdm

def reclassify_las(input_file, output_file):
    # 读取LAS文件
    las = laspy.read(input_file)
    
    # 获取当前的分类
    classification = las.classification
    
    # 创建一个映射字典
    class_map = {0: 1, 1: 2, 2: 3, 4: 4}
    
    # 应用映射
    new_classification = np.array([class_map.get(c, c) for c in tqdm(classification, desc="Reclassifying", unit='point')], dtype=np.uint8)
    
    # 更新分类
    las.classification = new_classification
    
    # 保存修改后的LAS文件
    las.write(output_file)
    
    print(f"Reclassification complete. Output saved to {output_file}")

# 使用
name="小454-458-labeled.las.clean"
input_file = f"No.2_小0454～小0458\\2_点群データ小454~458\\{name}.las"
output_file = f"Labeled\\{name}.las"
reclassify_las(input_file, output_file)

# 验证结果
original_las = laspy.read(input_file)
new_las = laspy.read(output_file)

print("Done!")

