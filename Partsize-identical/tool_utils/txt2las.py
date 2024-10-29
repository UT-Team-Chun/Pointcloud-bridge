import laspy
import numpy as np
import os
from glob import glob
from tqdm import tqdm

def txt_to_las(input_file, output_file):
    # 读取txt文件
    data = np.loadtxt(input_file)

    # 分离各列数据
    x, y, z, r, g, b, label = data.T

    # 将RGB值从0-1范围转换为0-65535范围
    r_16bit = (r * 65535).astype(np.uint16)
    g_16bit = (g * 65535).astype(np.uint16)
    b_16bit = (b * 65535).astype(np.uint16)

    # 创建LAS文件
    las = laspy.create(file_version="1.3", point_format=3)

    # 设置头部信息
    las.header.offsets = [np.min(x), np.min(y), np.min(z)]
    las.header.scales = [0.001, 0.001, 0.001]

    # 写入点数据
    las.x = x
    las.y = y
    las.z = z
    las.red = r_16bit
    las.green = g_16bit
    las.blue = b_16bit
    las.classification = label.astype(np.uint8)

    # 保存LAS文件
    las.write(output_file)

def process_folder(input_folder, output_folder):
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 获取所有txt文件
    txt_files = glob(os.path.join(input_folder, '*.txt'))
    total_files = len(txt_files)

    print(f"Find {total_files} txt file wait to precessing.")

    # 使用tqdm创建进度条
    for txt_file in tqdm(txt_files, desc="Processing progress", unit="file"):
        # 生成输出文件名
        base_name = os.path.basename(txt_file)
        output_file = os.path.join(output_folder, base_name.replace('.txt', '.las'))

        try:
            # 处理文件
            txt_to_las(txt_file, output_file)
        except Exception as e:
            print(f"处理文件 {txt_file} 时出错: {str(e)}")

    print(f"\n处理完成。共处理 {total_files} 个文件。")


# 使用示例
script_dir=os.getcwd()
# 更改当前工作目录
os.chdir(script_dir)
path=os.path.abspath(os.path.join(os.getcwd(), ".."))
input_folder = './Partsize-identical/log/sem_seg/2024-10-05_01-43/visual'  # 输入文件夹路径
output_folder = 'output_las'  # 输出文件夹路径

process_folder(input_folder, output_folder)

print("All files have been processed.")
