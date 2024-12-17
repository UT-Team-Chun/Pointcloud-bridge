import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib

matplotlib.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 11,
    'axes.labelsize': 11,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.dpi': 300
})


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
        'Metrics': ['MAE (mm)', 'RMSE (mm)', 'R²', 'Pearson r', 'RPE (%)'],
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
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

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
        yticks = np.linspace(-y_range, y_range, 11)  # 设置11个刻度点（包括0）
        ax.set_yticks(yticks)

        # x轴也类似处理
        x_range = mean.max() - mean.min()
        x_margin = x_range * 0.05  # 5%的边距
        ax.set_xlim(mean.min() - x_margin, mean.max() + x_margin)
        xticks = np.linspace(mean.min(), mean.max(), 11)  # 同样设置11个刻度点
        ax.set_xticks(xticks)

        # 设置刻度格式
        ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))  # 保持两位小数
        ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))

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
                   alpha=0.6,
                   edgecolor='black',
                   linewidth=0.5,
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
            f'n = {len(diff)}\n'
            f'Mean diff: {md:.2f}\n'
            f'SD: {sd:.2f}\n'
            f'95% CI of mean: [{ci_mean[0]:.2f}, {ci_mean[1]:.2f}]\n'
            f'CR: {(1.96 * 2 * sd):.2f}'  # Coefficient of Repeatability
        )

        ax.text(0.02, 0.98, stats_text,
                transform=ax.transAxes,
                verticalalignment='top',
                horizontalalignment='left',
                bbox=dict(boxstyle='round',
                          facecolor='white',
                          edgecolor='black',
                          alpha=0.9,
                          pad=0.5))

        # 设置标签和标题
        ax.set_xlabel(f'Mean of True and Predicted {dimension.capitalize()} (mm)',
                      fontsize=10)
        ax.set_ylabel('Difference (Predicted - True) (mm)',
                      fontsize=10)
        ax.set_title(f'{title} Bland-Altman Plot',
                     pad=15, fontsize=12)

        # 美化坐标轴
        ax.tick_params(width=1.25, length=6)

        # 设置轴线样式
        for spine in ax.spines.values():
            spine.set_linewidth(1.25)
            spine.set_color('black')

        # 设置网格样式
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.3)

        # 优化图例
        ax.legend(loc='upper right',
                  bbox_to_anchor=(0.98, 0.98),
                  frameon=True,
                  edgecolor='black',
                  framealpha=0.9,
                  fancybox=False,
                  borderaxespad=0.5,
                  fontsize=9)

        # 设置适当的y轴范围
        y_range = max(abs(diff.max()), abs(diff.min()))
        ax.set_ylim(-y_range * 1.1, y_range * 1.1)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.savefig(save_path.replace('.png', '.pdf'), format='pdf', bbox_inches='tight')

    return fig


def create_error_distribution_plot(df1, df2, dimension='length', save_path=None):
    """创建相对误差分布图

    Parameters:
        df1, df2: 包含真实值和预测值的数据框
        dimension: 测量维度 ('length', 'width', 等)
        save_path: 保存路径
    """
    from scipy import stats

    true_col = f'true_{dimension}'
    pred_col = f'pred_{dimension}'

    # 设置图形风格
    #plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # 颜色设置
    hist_color = '#2f5c85'  # 深蓝色
    kde_color = '#c44e52'  # 砖红色

    for ax, df, title in zip([ax1, ax2], [df1, df2], ['Dataset 1', 'Dataset 2']):
        # 计算相对误差（百分比）
        rel_error = (np.abs((df[pred_col] - df[true_col])) / df[true_col]) * 100

        # 计算统计指标
        mean_err = np.mean(rel_error)
        std_err = np.std(rel_error)
        median_err = np.median(rel_error)

        # 计算95%置信区间
        ci_lower, ci_upper = stats.norm.interval(0.95, loc=mean_err, scale=std_err / np.sqrt(len(rel_error)))

        # 绘制直方图和KDE
        sns.histplot(data=rel_error,
                     ax=ax,
                     color=hist_color,
                     alpha=0.6,
                     stat='density',  # 使用密度而不是计数
                     edgecolor='black',
                     linewidth=0.8)

        sns.kdeplot(data=rel_error,
                    ax=ax,
                    color=kde_color,
                    linewidth=2,
                    label='KDE')

        # 添加零线
        ax.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.5)

        # 设置标签和标题
        ax.set_xlabel('Relative Error (%)')
        ax.set_ylabel('Density')
        ax.set_title(f'{title} Error Distribution')

        # 添加详细的统计信息
        stats_text = (
            f'Mean: {mean_err:.2f}%\n'
            f'Median: {median_err:.2f}%\n'
            f'Std: {std_err:.2f}%\n'
            f'95% CI: [{ci_lower:.2f}, {ci_upper:.2f}]%\n'
            f'n = {len(rel_error)}'
        )

        # 添加统计文本框
        ax.text(0.95, 0.95, stats_text,
                transform=ax.transAxes,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round',
                          facecolor='white',
                          edgecolor='black',
                          alpha=0.9,
                          pad=0.5))

        # 美化坐标轴
        ax.tick_params(width=1.25, length=6)

        # 设置轴线样式
        for spine in ax.spines.values():
            spine.set_linewidth(1.25)
            spine.set_color('black')

        # 设置网格样式
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.3)

        # 设置合适的x轴范围（基于数据的分布）
        q1, q3 = np.percentile(rel_error, [25, 75])
        iqr = q3 - q1
        x_min = max(q1 - 3 * iqr, rel_error.min())
        x_max = min(q3 + 3 * iqr, rel_error.max())
        ax.set_xlim(x_min, x_max)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.savefig(save_path.replace('.png', '.pdf'), format='pdf', bbox_inches='tight')

    return fig


def plot_component_relative_errors(df1, df2, dimension='length', save_path=None):
    """绘制按构件分类的相对误差柱状图"""

    true_col = f'true_{dimension}'
    pred_col = f'pred_{dimension}'

    # 计算每个构件的相对误差
    components = ['Deck', 'Girder', 'Parapet']
    errors_df1 = []
    errors_df2 = []

    for comp in components:
        # Dataset 1
        mask1 = df1['component'] == comp
        error1 = calculate_relative_error(
            df1[mask1][true_col],
            df1[mask1][pred_col]
        ).mean()
        errors_df1.append(error1)

        # Dataset 2
        mask2 = df2['component'] == comp
        error2 = calculate_relative_error(
            df2[mask2][true_col],
            df2[mask2][pred_col]
        ).mean()
        errors_df2.append(error2)

    fig, ax = plt.subplots(figsize=(8, 6))

    # 设置网格
    ax.grid(True, linestyle='-.', linewidth=0.5, color='gray', alpha=0.5)

    x = np.arange(len(components))
    width = 0.35

    # 柱状图部分
    rects1 = ax.bar(x - width / 2, errors_df1, width, label='Dataset 1',
                    color='#8dd3c7', edgecolor='black', linewidth=0.75)
    rects2 = ax.bar(x + width / 2, errors_df2, width, label='Dataset 2',
                    color='#bebada', edgecolor='black', linewidth=0.75)

    # 平均线部分 - 加深颜色
    mean1 = np.mean(errors_df1)
    mean2 = np.mean(errors_df2)
    ax.axhline(mean1, color='#5bb3a7', linestyle='--', linewidth=1.5)  # 加深的绿色
    ax.axhline(mean2, color='#9281c9', linestyle='--', linewidth=1.5)  # 加深的紫色

    # 添加数值标注
    def autolabel(rects, values):
        for rect, val in zip(rects, values):
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2., height,
                    f'{val:.1f}%',
                    ha='center', va='bottom',
                    fontsize=9, fontweight='bold')

    autolabel(rects1, errors_df1)
    autolabel(rects2, errors_df2)

    # 添加平均值标注
    ax.text(ax.get_xlim()[1], mean1, f'Mean: {mean1:.1f}%',
            ha='right', va='bottom', color='#5bb3a7',
            fontsize=9, fontweight='bold')
    ax.text(ax.get_xlim()[1], mean2, f'Mean: {mean2:.1f}%',
            ha='right', va='bottom', color='#9281c9',
            fontsize=9, fontweight='bold')

    ax.set_ylabel('Relative Error (%)', fontsize=10)
    ax.set_title(f'Relative Errors by Component ({dimension})', fontsize=11, pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(components, fontsize=10)

    # 设置刻度线样式
    ax.tick_params(width=1.25, length=6)

    # 设置轴线样式
    for spine in ax.spines.values():
        spine.set_linewidth(1.25)
        spine.set_color('black')

    ax.legend(loc='upper right', bbox_to_anchor=(0.98, 0.98),
              frameon=True, edgecolor='black', framealpha=1.0,
              fancybox=False, borderaxespad=0.5)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.savefig(save_path.replace('.png', '.pdf'), format='pdf', bbox_inches='tight')

    return fig


def plot_case_errors(df1, df2, dimension='length', save_path=None):
    """绘制按case分类的误差图"""
    true_col = f'true_{dimension}'
    pred_col = f'pred_{dimension}'

    # 定义两个数据集的cases
    cases_df1 = ['b1', 'b2', 'b7']
    cases_df2 = ['cb2', 'cb6', 'cb9']

    # 创建单个图
    fig, ax = plt.subplots(figsize=(10, 6))
    # 设置样式
    #plt.style.use("seaborn-v0_8-paper")

    # 准备数据
    errors_df1 = []
    errors_df2 = []

    # Dataset 1数据
    for case in cases_df1:
        mask = df1['case'] == case
        error = calculate_relative_error(
            df1[mask][true_col],
            df1[mask][pred_col]
        )
        errors_df1.append(error)

    # Dataset 2数据
    for case in cases_df2:
        mask = df2['case'] == case
        error = calculate_relative_error(
            df2[mask][true_col],
            df2[mask][pred_col]
        )
        errors_df2.append(error)

    # 设置位置
    positions = np.arange(1, len(cases_df1) + len(cases_df2) + 1)
    # 设置网格
    ax.grid(True, linestyle='-.', linewidth=0.5, color='gray', alpha=0.5)

    # 绘制箱线图
    bp1 = ax.boxplot(errors_df1, positions=positions[:len(cases_df1)],
                     patch_artist=True,
                     boxprops=dict(facecolor='lightblue', color='blue'),
                     medianprops=dict(color='blue'),
                     whiskerprops=dict(color='blue'),
                     capprops=dict(color='blue'),
                     flierprops=dict(color='blue', markeredgecolor='blue'))

    bp2 = ax.boxplot(errors_df2, positions=positions[len(cases_df1):],
                     patch_artist=True,
                     boxprops=dict(facecolor='lightgreen', color='green'),
                     medianprops=dict(color='green'),
                     whiskerprops=dict(color='green'),
                     capprops=dict(color='green'),
                     flierprops=dict(color='green', markeredgecolor='green'))

    # 设置轴线样式
    for spine in ax.spines.values():
        spine.set_linewidth(1.25)
        spine.set_color('black')

    # 设置刻度线样式
    ax.tick_params(width=1.25, length=6)

    # 设置标签和标题
    ax.set_xticks(positions)
    ax.set_xticklabels(cases_df1 + cases_df2)
    ax.set_ylabel('Relative Error (%)')
    ax.set_xlabel('Case')
    ax.set_title(f'{dimension.capitalize()} Errors by Case')

    # 添加图例
    ax.plot([], [], color='black', marker='s', markerfacecolor='lightblue',
            label='Dataset 1', linestyle='', markersize=10, markeredgewidth=0.75)
    ax.plot([], [], color='black', marker='s', markerfacecolor='lightgreen',
            label='Dataset 2', linestyle='', markersize=10, markeredgewidth=0.75)

    ax.legend(loc='upper left', bbox_to_anchor=(0.02, 0.98),
              frameon=True, edgecolor='black', framealpha=1.0,
              fancybox=False, borderaxespad=0.5)

    # 添加分隔线
    ax.axvline(x=len(cases_df1) + 0.5, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.savefig(save_path.replace('.png', '.pdf'), format='pdf', bbox_inches='tight')

    return fig


def plot_regression_analysis(df1, df2, dimension='length', save_path=None):
    """绘制回归分析图"""
    true_col = f'true_{dimension}'
    pred_col = f'pred_{dimension}'

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # 第一组数据
    sns.regplot(data=df1, x=true_col, y=pred_col, ax=ax1,
                scatter_kws={'alpha': 0.6,
                             'color': '#2f5c85',  # 填充色
                             'edgecolor': 'black',  # 黑色边框
                             'marker': 'D',  # 菱形
                             's': 50,  # 稍微调小一点以适应菱形
                             'linewidths': 0.75},  #

                line_kws={'color': '#c44e52',    # 砖红色
                         'label': 'Regression line',
                         'linewidth': 1.5},
                ci=95)
    ax1.fill_between([], [], [], color='#c44e52', alpha=0.15, label='95% CI')
    ax1.set_title('Dataset 1 Regression Analysis')

    # 第二组数据
    sns.regplot(data=df2, x=true_col, y=pred_col, ax=ax2,
                scatter_kws={'alpha': 0.6,
                             'color': '#2f5c85',  # 填充色
                             'edgecolor': 'black',  # 黑色边框
                             'marker': 'D',  # 菱形
                             's': 50,  # 稍微调小一点以适应菱形
                             'linewidths': 0.75},  # 注意这里改为 linewidths
                line_kws={'color': '#c44e52',
                         'label': 'Regression line',
                         'linewidth': 1.5},
                ci=95)
    ax2.fill_between([], [], [], color='#c44e52', alpha=0.15, label='95% CI')
    ax2.set_title('Dataset 2 Regression Analysis')

    for ax in [ax1, ax2]:
        ax.set_xlabel(f'True {dimension.capitalize()} (mm)')
        ax.set_ylabel(f'Predicted {dimension.capitalize()} (mm)')
        lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),
            np.max([ax.get_xlim(), ax.get_ylim()]),
        ]
        ax.plot(lims, lims, '--', color='#555555', alpha=0.6,
                label='Identity line', linewidth=1.25)

        # 设置刻度线样式
        ax.tick_params(width=1.25, length=6)

        # 设置轴线样式
        for spine in ax.spines.values():
            spine.set_linewidth(1.25)
            spine.set_color('black')

        ax.legend(loc='upper left', bbox_to_anchor=(0.02, 0.98),
                  frameon=True, edgecolor='black', framealpha=1.0,
                  fancybox=False, borderaxespad=0.5)

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

    # 绘制图表
    plot_regression_analysis(df1, df2, 'length', 'results/length_regression.png')
    plot_regression_analysis(df1, df2, 'width', 'results/width_regression.png')

    plot_bland_altman(df1, df2, 'length', 'results/length_bland_altman.png')
    plot_bland_altman(df1, df2, 'width', 'results/width_bland_altman.png')

    create_error_distribution_plot(df1, df2, 'length', 'results/length_error_dist.png')
    create_error_distribution_plot(df1, df2, 'width', 'results/width_error_dist.png')

    # 新增分析
    plot_component_relative_errors(df1, df2, 'length', 'results/length_component_errors.png')
    plot_component_relative_errors(df1, df2, 'width', 'results/width_component_errors.png')

    plot_case_errors(df1, df2, 'length', 'results/length_case_errors.png')
    plot_case_errors(df1, df2, 'width', 'results/width_case_errors.png')

    print('Analysis completed!')


if __name__ == '__main__':
    main()
