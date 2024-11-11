import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path

# 获取当前文件的路径
current_file = Path(__file__)
print(f"当前文件路径：{current_file}")

# 获取项目根目录（向上两级）
root_dir = current_file.parent.parent
print(f"项目根目录：{root_dir}")

data_dir = root_dir / 'evaluation_results.csv'
# 读取数据
df = pd.read_csv(data_dir)


# 1. 计算基本统计指标
length_error = np.abs(df['Original Length'] - df['Estimated Length']) / df['Original Length']
width_error = np.abs(df['Original Width'] - df['Estimated Width']) / df['Original Width']

print("统计分析结果：")
print("\n1. 相对误差统计：")
print(f"平均相对误差: {df['Relative Error'].mean():.4f}")
print(f"相对误差标准差: {df['Relative Error'].std():.4f}")
print(f"最大相对误差: {df['Relative Error'].max():.4f}")
print(f"最小相对误差: {df['Relative Error'].min():.4f}")

# 2. 进行可视化分析
plt.figure(figsize=(15, 10))

# 2.1 长度对比散点图
plt.subplot(221)
plt.scatter(df['Original Length'], df['Estimated Length'], c=df['Label'], cmap='viridis')
plt.plot([10, 20], [10, 20], 'r--')  # 理想预测线
plt.xlabel('Original Length')
plt.ylabel('Estimated Length')
plt.title('Original vs Estimated Length')

# 2.2 宽度对比散点图
plt.subplot(222)
plt.scatter(df['Original Width'], df['Estimated Width'], c=df['Label'], cmap='viridis')
plt.plot([2, 10], [2, 10], 'r--')  # 理想预测线
plt.xlabel('Original Width')
plt.ylabel('Estimated Width')
plt.title('Original vs Estimated Width')

# 2.3 相对误差箱型图
plt.subplot(223)
sns.boxplot(data=df, x='Label', y='Relative Error')
plt.title('Relative Error Distribution by Label')

# 2.4 误差分布直方图
plt.subplot(224)
plt.hist(df['Relative Error'], bins=10, edgecolor='black')
plt.xlabel('Relative Error')
plt.ylabel('Frequency')
plt.title('Distribution of Relative Error')

plt.tight_layout()
plt.show()

# 3. 计算每个Label的平均误差
label_stats = df.groupby('Label')['Relative Error'].agg(['mean', 'std']).round(4)
print("\n2. 不同Label的误差统计：")
print(label_stats)

# 4. 进行相关性分析
correlation_matrix = df[['Original Length', 'Original Width', 'Estimated Length', 'Estimated Width']].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Matrix')
plt.show()
