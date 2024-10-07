import os

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 设置文件夹路径
folder_path = '.'

# 获取文件夹中所有的CSV文件
csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

if not csv_files:
    print("没有找到CSV文件。创建一个示例CSV文件用于演示。")
    # 创建一个示例CSV文件
    df = pd.DataFrame({
        'x': np.arange(0, 10),
        'y': np.random.rand(10)
    })
    df.to_csv(os.path.join(folder_path, 'example.csv'), index=False)
    csv_files = ['example.csv']

# 读取第一个CSV文件（如果有多个文件的话）
file_path = os.path.join(folder_path, csv_files[0])
df = pd.read_csv(file_path)

print(f"读取的CSV文件: {csv_files[0]}")
print("数据预览:")
print(df.head())

# 使用matplotlib绘制简单的折线图
plt.figure(figsize=(10, 6))
plt.plot(df.iloc[:, 0], df.iloc[:, 1], marker='o')
plt.title(f'CSV数据可视化 - {csv_files[0]}')
plt.xlabel(df.columns[0])
plt.ylabel(df.columns[1])
plt.grid(True)
plt.show()