import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import mean_squared_error, r2_score
#import statsmodels.api as sm
from mplfonts import use_font
#use_font('Noto Serif CJK SC')

def load_and_process_data(file1, file2):
    """加载并预处理两组数据"""
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    
    # 重命名列以确保统一性
    columns = ['case', 'component', 'true_length', 'pred_length', 
              'true_width', 'pred_width']
    df1.columns = columns
    df2.columns = columns
    
    return df1, df2

def calculate_metrics(true_vals, pred_vals):
    """计算各种评价指标"""
    metrics = {
        'MAE': np.mean(np.abs(true_vals - pred_vals)),
        'RMSE': np.sqrt(mean_squared_error(true_vals, pred_vals)),
        'R2': r2_score(true_vals, pred_vals),
        'Pearson_r': stats.pearsonr(true_vals, pred_vals)[0],
        'RPE': np.mean(np.abs(true_vals - pred_vals) / true_vals) * 100
    }
    return metrics

def create_comparison_table(df1, df2, dimension='length'):
    """创建比较表格"""
    true_col = f'true_{dimension}'
    pred_col = f'pred_{dimension}'
    
    metrics1 = calculate_metrics(df1[true_col], df1[pred_col])
    metrics2 = calculate_metrics(df2[true_col], df2[pred_col])
    
    # 合并两组指标
    comparison = pd.DataFrame({
        'Metrics': ['MAE (mm)', 'RMSE (mm)', 'R²', 'Pearson r', 'RPE (%)'],
        'Group 1': [metrics1[k] for k in ['MAE', 'RMSE', 'R2', 'Pearson_r', 'RPE']],
        'Group 2': [metrics2[k] for k in ['MAE', 'RMSE', 'R2', 'Pearson_r', 'RPE']]
    })
    
    return comparison

def plot_regression_analysis(df1, df2, dimension='length', save_path=None):
    """绘制回归分析图"""
    true_col = f'true_{dimension}'
    pred_col = f'pred_{dimension}'
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # 设置样式
    plt.style.use("seaborn-v0_8-whitegrid")
    
    # 第一组数据
    sns.regplot(data=df1, x=true_col, y=pred_col, ax=ax1,
                scatter_kws={'alpha':0.5}, line_kws={'color': 'red'})
    ax1.set_title('Group 1 Regression Analysis', fontsize=12)
    
    # 第二组数据
    sns.regplot(data=df2, x=true_col, y=pred_col, ax=ax2,
                scatter_kws={'alpha':0.5}, line_kws={'color': 'red'})
    ax2.set_title('Group 2 Regression Analysis', fontsize=12)
    
    # 添加标签
    for ax in [ax1, ax2]:
        ax.set_xlabel(f'True {dimension.capitalize()} (mm)', fontsize=10)
        ax.set_ylabel(f'Predicted {dimension.capitalize()} (mm)', fontsize=10)
        
        # 添加对角线
        lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),
            np.max([ax.get_xlim(), ax.get_ylim()]),
        ]
        ax.plot(lims, lims, '--', color='gray', alpha=0.75)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def plot_bland_altman(df1, df2, dimension='length', save_path=None):
    """绘制Bland-Altman图"""
    true_col = f'true_{dimension}'
    pred_col = f'pred_{dimension}'
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    for ax, df, title in zip([ax1, ax2], [df1, df2], ['Group 1', 'Group 2']):
        mean = (df[true_col] + df[pred_col]) / 2
        diff = df[pred_col] - df[true_col]
        
        md = np.mean(diff)
        sd = np.std(diff)
        
        # 绘制散点图
        ax.scatter(mean, diff, alpha=0.5)
        
        # 添加平均差异线和95%置信区间
        ax.axhline(md, color='red', linestyle='-', label='Mean difference')
        ax.axhline(md + 1.96*sd, color='gray', linestyle='--', 
                  label='95% limits of agreement')
        ax.axhline(md - 1.96*sd, color='gray', linestyle='--')
        
        ax.set_xlabel(f'Mean of True and Predicted {dimension.capitalize()} (mm)')
        ax.set_ylabel('Difference (Predicted - True) (mm)')
        ax.set_title(f'{title} Bland-Altman Plot')
        ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def create_error_distribution_plot(df1, df2, dimension='length', save_path=None):
    """创建误差分布图"""
    true_col = f'true_{dimension}'
    pred_col = f'pred_{dimension}'
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    for ax, df, title in zip([ax1, ax2], [df1, df2], ['Group 1', 'Group 2']):
        error = df[pred_col] - df[true_col]
        
        # 绘制误差分布
        sns.histplot(error, kde=True, ax=ax)
        ax.set_xlabel('Error (mm)')
        ax.set_ylabel('Count')
        ax.set_title(f'{title} Error Distribution')
        
        # 添加统计信息
        stats_text = f'Mean: {np.mean(error):.2f}\nStd: {np.std(error):.2f}'
        ax.text(0.95, 0.95, stats_text,
                transform=ax.transAxes,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def main():
    # 加载数据
    df1, df2 = load_and_process_data('bridge-set-1.csv', 'bridge-set-2.csv')
    
    # 创建结果文件夹
    import os
    if not os.path.exists('results'):
        os.makedirs('results')
    
    # 分析长度
    length_comparison = create_comparison_table(df1, df2, 'length')
    length_comparison.to_csv('results/length_metrics.csv', index=False)
    
    # 分析宽度
    width_comparison = create_comparison_table(df1, df2, 'width')
    width_comparison.to_csv('results/width_metrics.csv', index=False)
    
    # 绘制图表
    # 回归分析
    plot_regression_analysis(df1, df2, 'length', 'results/length_regression.png')
    plot_regression_analysis(df1, df2, 'width', 'results/width_regression.png')
    
    # Bland-Altman分析
    plot_bland_altman(df1, df2, 'length', 'results/length_bland_altman.png')
    plot_bland_altman(df1, df2, 'width', 'results/width_bland_altman.png')
    
    # 误差分布
    create_error_distribution_plot(df1, df2, 'length', 
                                 'results/length_error_dist.png')
    create_error_distribution_plot(df1, df2, 'width', 
                                 'results/width_error_dist.png')
    
    print('Analysis completed!')

if __name__ == '__main__':
    main()
