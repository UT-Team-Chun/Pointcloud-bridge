import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
from sklearn.metrics import (
    mean_squared_error,  # MSE
    mean_absolute_error,  # MAE
    r2_score, # R2
)

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


y_l = df['Original Length']
y_lhat = df['Estimated Length']
y_w = df['Original Width']
y_what = df['Estimated Width']

y_t = np.concatenate((df['Original Length'], df['Original Width']))
y_that = np.concatenate((df['Estimated Length'], df['Estimated Width']))

def model_evl(y,y_hat):
    mse_skl = mean_squared_error(y, y_hat)
    rmse_skl = np.sqrt(mse_skl)
    mae_skl = mean_absolute_error(y, y_hat)
    mape_skl = 100. * mean_absolute_error(np.ones_like(y), y_hat / y)
    r2_skl = r2_score(y, y_hat)

    print(f'MSE (sklearn): {mse_skl}.')
    print(f'RMSE (sklearn): {rmse_skl}.')
    print(f'MAE (sklearn): {mae_skl}.')
    print(f'MAPE (sklearn): {mape_skl}.')
    print(f'R2 (sklearn): {r2_skl}.')

model_evl(y_l,y_lhat)
model_evl(y_w,y_what)
model_evl(y_t,y_that)

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
correlation_matrix = df[['Original Length', 'Estimated Width']].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Matrix')
plt.show()
