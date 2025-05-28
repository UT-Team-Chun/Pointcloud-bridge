import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.ticker


def set_scientific_style():
    plt.style.use('default')  # 首先重置为默认样式

    matplotlib.rcParams.update({
        # 字体设置
        'font.family': 'Arial',  # 使用 Arial 字体
        'font.size': 24,
        'axes.titlesize': 26,
        'axes.labelsize': 26,
        'xtick.labelsize': 24,
        'ytick.labelsize': 24,
        'legend.fontsize': 22,

        # 线条设置
        'axes.linewidth': 1.25,  # 坐标轴线宽
        'grid.linewidth': 0.5,  # 网格线宽
        'lines.linewidth': 1.5,  # 曲线线宽
        'patch.linewidth': 0.75,  # 填充区域边界线宽

        # 刻度设置
        'xtick.major.width': 1.25,
        'ytick.major.width': 1.25,
        'xtick.minor.width': 1.0,
        'ytick.minor.width': 1.0,
        'xtick.major.size': 6,
        'ytick.major.size': 6,
        'xtick.minor.size': 4,
        'ytick.minor.size': 4,

        # 图例设置
        'legend.frameon': True,
        'legend.framealpha': 1.0,
        'legend.edgecolor': 'black',
        'legend.fancybox': False,

        # 其他设置
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,

        # 颜色设置
        'axes.prop_cycle': plt.cycler(color=['#2f5c85', '#c44e52',
                                             '#8dd3c7', '#bebada',
                                             '#5bb3a7', '#9281c9'])
    })
# matplotlib.rcParams.update({
#     'font.size': 11,
#     'axes.titlesize': 11,
#     'axes.labelsize': 11,
#     'xtick.labelsize': 11,
#     'ytick.labelsize': 11,
#     'legend.fontsize': 11,
#     'figure.dpi': 300
# })


def load_and_process_data(file1, file2):
    """加载并预处理两组数据"""
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

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

    comparison = pd.DataFrame({
        'Metrics': ['MAE (m)', 'RMSE (m)', 'R²', 'Pearson r', 'RPE (%)'],
        'Dataset 1': [metrics1[k] for k in ['MAE', 'RMSE', 'R2', 'Pearson_r', 'RPE']],
        'Dataset 2': [metrics2[k] for k in ['MAE', 'RMSE', 'R2', 'Pearson_r', 'RPE']]
    })

    return comparison


def calculate_relative_error(true_val, pred_val):
    """计算相对误差"""
    return np.abs(true_val - pred_val) / true_val * 100


def plot_bland_altman(df1, df2, dimension='length', save_path=None):
    """绘制改进的Bland-Altman图

    Parameters:
        df1, df2: 包含真实值和预测值的数据框
        dimension: 测量维度 ('length', 'width', 等)
        save_path: 保存路径
    """
    from scipy import stats

    true_col = f'true_{dimension}'
    pred_col = f'pred_{dimension}'

    # 设置更大的图形尺寸
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # 设置颜色方案
    scatter_color = '#2f5c85'  # 深蓝色
    mean_line_color = '#c44e52'  # 砖红色
    limit_line_color = '#7a7a7a'  # 深灰色

    for ax, df, title in zip([ax1, ax2], [df1, df2], ['Dataset 1', 'Dataset 2']):
        mean = (df[true_col] + df[pred_col]) / 2
        diff = df[pred_col] - df[true_col]

        # 计算统计量
        md = np.mean(diff)
        sd = np.std(diff)
        # 设置y轴范围和刻度
        y_range = max(abs(diff.max()), abs(diff.min())) * 1.1  # 使用1.1倍的最大范围
        ax.set_ylim(-y_range, y_range)

        # 设置合适数量的刻度（比如10个左右）
        yticks = np.linspace(-y_range, y_range, 9)  # 设置11个刻度点（包括0）
        ax.set_yticks(yticks)

        # x轴也类似处理
        x_range = mean.max() - mean.min()
        x_margin = x_range * 0.05  # 5%的边距
        ax.set_xlim(mean.min() - x_margin, mean.max() + x_margin)
        ax.tick_params(axis='x', rotation=45)

        # 设置刻度格式
        ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.1f'))
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=9, prune='both'))
        ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.1f'))

        # 计算95%置信区间
        ci_mean = stats.t.interval(confidence=0.95,
                                   df=len(diff) - 1,
                                   loc=md,
                                   scale=stats.sem(diff))

        # 计算一致性限值的95%置信区间
        loa_upper = md + 1.96 * sd
        loa_lower = md - 1.96 * sd

        # 计算LoA的标准误
        se_loa = np.sqrt(3 * sd ** 2 / len(diff))
        ci_upper = stats.t.interval(confidence=0.95,
                                    df=len(diff) - 1,
                                    loc=loa_upper,
                                    scale=se_loa)
        ci_lower = stats.t.interval(confidence=0.95,
                                    df=len(diff) - 1,
                                    loc=loa_lower,
                                    scale=se_loa)

        # 绘制散点图（添加边框）
        ax.scatter(mean, diff,
                   color=scatter_color,
                   alpha=0.8,
                   edgecolor='black',
                   linewidth=0.8,
                   s=120)

        # 添加平均差异线和95%置信区间
        ax.axhline(md, color=mean_line_color, linestyle='-', linewidth=2,
                   label=f'Mean difference: {md:.2f}')
        ax.axhline(loa_upper, color=limit_line_color, linestyle='--', linewidth=1.5,
                   label=f'95% LoA: [{loa_lower:.2f}, {loa_upper:.2f}]')
        ax.axhline(loa_lower, color=limit_line_color, linestyle='--', linewidth=1.5)

        # 添加置信区间阴影
        ax.fill_between([ax.get_xlim()[0], ax.get_xlim()[1]],
                        ci_mean[0], ci_mean[1],
                        color=mean_line_color, alpha=0.1)
        ax.fill_between([ax.get_xlim()[0], ax.get_xlim()[1]],
                        ci_upper[0], ci_upper[1],
                        color=limit_line_color, alpha=0.1)
        ax.fill_between([ax.get_xlim()[0], ax.get_xlim()[1]],
                        ci_lower[0], ci_lower[1],
                        color=limit_line_color, alpha=0.1)

        # 添加统计信息文本框
        stats_text = (
            #f'n = {len(diff)}; Mean diff: {md:.2f}\n'
           #f'Mean diff: {md:.2f}\n'
            f'SD: {sd:.2f}\n'
            f'CR: {(1.96 * 2 * sd):.2f} \n'  # Coefficient of Repeatability
            f'95% CI of mean: \n   [{ci_mean[0]:.2f}, {ci_mean[1]:.2f}]'
            #f'CR: {(1.96 * 2 * sd):.2f}'  # Coefficient of Repeatability
        )

        ax.text(0.03, 0.95, stats_text,
                transform=ax.transAxes,
                verticalalignment='top',
                horizontalalignment='left',
                fontsize=18,
                bbox=dict(boxstyle='round',
                          facecolor='white',
                          edgecolor='black',
                          alpha=0.6,
                          pad=0.5))

        # 设置标签和标题
        ax.set_xlabel(f'Mean of True and Predicted {dimension.capitalize()} (m)')
        ax.set_ylabel('Difference (Predicted - True) (m)')
        ax.set_title(f'{title} Bland-Altman Plot',
                     pad=15)

        # 美化坐标轴
        ax.tick_params(width=1.25, length=6)

        # 设置轴线样式
        for spine in ax.spines.values():
            spine.set_linewidth(1.25)
            spine.set_color('black')

        # 设置网格样式
        ax.grid(True, linestyle='--', linewidth=0.6, alpha=0.4)

        # 优化图例
        ax.legend(loc='upper right',
                  bbox_to_anchor=(0.99, 0.99),
                  frameon=True,
                  edgecolor='black',
                  framealpha=0.8,
                  fancybox=True,
                  borderaxespad=0.5,
                  fontsize=18)
                  
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.savefig(save_path.replace('.png', '.pdf'), format='pdf', bbox_inches='tight')

    return fig



def main():
    # 加载数据
    df1, df2 = load_and_process_data('bridge-set-1.csv', 'bridge-set-2.csv')

    # 创建结果文件夹
    import os
    if not os.path.exists('results'):
        os.makedirs('results')

    # 原有分析
    length_comparison = create_comparison_table(df1, df2, 'length')
    length_comparison.to_csv('results/length_metrics.csv', index=False)

    width_comparison = create_comparison_table(df1, df2, 'width')
    width_comparison.to_csv('results/width_metrics.csv', index=False)

    plot_bland_altman(df1, df2, 'length', 'results/length_bland_altman.png')
    plot_bland_altman(df1, df2, 'width', 'results/width_bland_altman.png')

    print('Analysis completed!')


if __name__ == '__main__':
    set_scientific_style()
    main()
