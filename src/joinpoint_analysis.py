import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import linregress
from main import process_data
from trend_analysis import calculate_standardized_rate, calculate_asr_mir

def find_joinpoints(x, y, n_joinpoints=2):
    """寻找最优拐点位置"""
    def objective(points):
        # 将点排序
        sorted_points = np.sort(points)
        segments_x = np.split(x, np.searchsorted(x, sorted_points))
        segments_y = np.split(y, np.searchsorted(x, sorted_points))
        
        # 计算总残差平方和
        total_rss = 0
        for seg_x, seg_y in zip(segments_x, segments_y):
            if len(seg_x) > 1:
                slope, intercept = np.polyfit(seg_x, seg_y, 1)
                y_pred = slope * seg_x + intercept
                total_rss += np.sum((seg_y - y_pred) ** 2)
        return total_rss

    # 基于数据特征的智能初始猜测
    # 计算一阶差分来找到可能的变化点
    diff_y = np.diff(y)
    abs_diff = np.abs(diff_y)
    
    # 找到变化最大的点作为候选拐点
    if len(abs_diff) >= n_joinpoints:
        # 找到变化最大的n_joinpoints个点
        candidate_indices = np.argsort(abs_diff)[-n_joinpoints:]
        candidate_indices = np.sort(candidate_indices)
        initial_points = x[candidate_indices + 1]  # +1因为diff比原数组少一个元素
    else:
        # 如果数据点不够，使用改进的均匀分布
        x_range = x.max() - x.min()
        # 添加随机扰动避免相同的初始点
        random_offset = np.random.uniform(-0.1, 0.1, n_joinpoints)
        base_positions = np.linspace(0.2, 0.8, n_joinpoints) + random_offset
        base_positions = np.clip(base_positions, 0.1, 0.9)  # 确保在合理范围内
        initial_points = x.min() + x_range * base_positions
    
    # 确保初始点在有效范围内
    initial_points = np.clip(initial_points, x.min() + 1, x.max() - 1)
    
    # 多次优化以避免局部最优解
    best_result = None
    best_rss = float('inf')
    
    for attempt in range(5):  # 尝试5次不同的初始化
        if attempt > 0:
            # 后续尝试使用随机初始化
            x_range = x.max() - x.min()
            random_positions = np.random.uniform(0.2, 0.8, n_joinpoints)
            current_initial = x.min() + x_range * random_positions
            current_initial = np.clip(current_initial, x.min() + 1, x.max() - 1)
        else:
            current_initial = initial_points
        
        bounds = [(x.min() + 1, x.max() - 1)] * n_joinpoints
        try:
            result = minimize(objective, current_initial, bounds=bounds, method='L-BFGS-B')
            if result.success and result.fun < best_rss:
                best_rss = result.fun
                best_result = result
        except:
            continue
    
    if best_result is None:
        # 如果所有优化都失败，返回简单的均匀分布
        x_range = x.max() - x.min()
        return x.min() + x_range * np.linspace(0.3, 0.7, n_joinpoints)
    
    return np.sort(best_result.x)

def calculate_slopes(x, y, joinpoints):
    """计算各段斜率"""
    segments_x = np.split(x, np.searchsorted(x, joinpoints))
    segments_y = np.split(y, np.searchsorted(x, joinpoints))
    
    slopes = []
    for seg_x, seg_y in zip(segments_x, segments_y):
        if len(seg_x) > 1:
            slope, _, _, _, _ = linregress(seg_x, seg_y)
            slopes.append(slope)
    return slopes

def plot_joinpoint_analysis(df_incidence, df_mortality, df_population):

    # 创建6×3的子图布局
    fig, axes = plt.subplots(6, 3, figsize=(27, 34))
    
    locations = ['China', 'G20']
    years = list(range(1990, 2022))

    titles = ['ASIR', 'ASMR', 'ASR of MIR']
    sexes = ['Both', 'Male', 'Female']
    
    # 处理每个地区的数据
    for loc_idx, location in enumerate(locations):
        
        # 处理数据
        incidence_data = df_incidence[
            (df_incidence['measure_name'] == 'Incidence') & 
            (df_incidence['metric_name'] == 'Number') &
            (df_incidence['location_name'] == location)
        ].copy()
        
        mortality_data = df_mortality[
            (df_mortality['measure_name'] == 'Deaths') & 
            (df_incidence['metric_name'] == 'Number') &
            (df_mortality['location_name'] == location)
        ].copy()
        
        
        # 获取所有年龄组的数据
        all_ages_data = {}
        for sex in sexes:
            all_ages_data[sex] = {
                'years': years,
                'asir': [],
                'asmr': [],
                'asr_mir': []
            }
            
            for year in years:
                # 获取当年的发病和死亡数据
                year_inc = incidence_data[
                    (incidence_data['year'] == year) & 
                    (incidence_data['age_name'].isin(['5-14 years', '15-49 years', '50-69 years', '70+ years']))
                ]
                year_mort = mortality_data[
                    (mortality_data['year'] == year) & 
                    (mortality_data['age_name'].isin(['5-14 years', '15-49 years', '50-69 years', '70+ years']))
                ]
                
                # 获取对应的人口数据，并按性别筛选
                year_pop = df_population[
                    (df_population['location'] == location) & 
                    (df_population['year'] == year) &
                    (df_population['age'].isin(['5-14 years', '15-49 years', '50-69 years', '70+ years'])) &
                    (df_population['sex'] == sex)  # 添加性别筛选
                ].copy()
                
                # 确保每个年龄组只有一个值
                year_pop = year_pop.groupby('age')['val'].sum().reset_index()
                
               
                sex_inc = year_inc[year_inc['sex_name'] == sex].copy()
                sex_mort = year_mort[year_mort['sex_name'] == sex].copy()
                
                if sex_inc.empty or sex_mort.empty:
                    print(f"Warning: Empty data for {location}, {sex}, {year}")
                    continue
                
                try:
                    # 使用 age 列进行映射
                    sex_inc['population'] = sex_inc['age_name'].map(
                        year_pop.set_index('age')['val']
                    )
                    sex_mort['population'] = sex_mort['age_name'].map(
                        year_pop.set_index('age')['val']
                    )
                    asir = calculate_standardized_rate(sex_inc, year_pop, 'ASIR')
                    asmr = calculate_standardized_rate(sex_mort, year_pop, 'ASMR')
                    asr_mir = calculate_asr_mir(asir, asmr)
                    
                    if year % 5 == 0:  # 每5年打印一次计算结果
                        print(f"Rates for {year}: ASIR={asir:.4f}, ASMR={asmr:.4f}, MIR={asr_mir:.4f}")
                    
                except Exception as e:
                    print(f"Error processing {location}, {sex}, {year}: {str(e)}")
                    continue
                
                all_ages_data[sex]['asir'].append(asir)
                all_ages_data[sex]['asmr'].append(asmr)
                all_ages_data[sex]['asr_mir'].append(asr_mir)
        
        # 绘制趋势图和拐点
        start_idx = loc_idx * 3  # 每个地区占3行
        for sex_idx, sex in enumerate(sexes):
            row_idx = start_idx + sex_idx
            for col_idx, (title, measure) in enumerate(zip(titles, ['asir', 'asmr', 'asr_mir'])):
                ax = axes[row_idx, col_idx]
                
                # 确保数据点按年份排序并移除 NaN 值
                years_array = np.array(all_ages_data[sex]['years'])
                measure_array = np.array(all_ages_data[sex][measure])
                
                # 移除包含 NaN 的数据点
                valid_indices = ~np.isnan(measure_array)
                years_array = years_array[valid_indices]
                measure_array = measure_array[valid_indices]
                
                # 按年份排序
                sort_idx = np.argsort(years_array)
                years_array = years_array[sort_idx]
                measure_array = measure_array[sort_idx]
                
                # 寻找拐点
                joinpoints = find_joinpoints(years_array, measure_array, n_joinpoints=5)
                print(1, years_array)
                print(2, measure_array)
                print(3, joinpoints)
                slopes = calculate_slopes(years_array, measure_array, joinpoints)
                
                # 绘制数据点
                ax.scatter(years_array, measure_array, color='black', alpha=0.5, s=30)
                
                # 绘制连接线段（直接连接拐点，不进行拟合）
                segments_x = np.split(years_array, np.searchsorted(years_array, joinpoints))
                segments_y = np.split(measure_array, np.searchsorted(years_array, joinpoints))
                
                # 直接连接各个时间段的数据点
                colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'cyan']  # 为不同时期设置不同颜色
                # for i, (seg_x, seg_y) in enumerate(zip(segments_x, segments_y)):
                    # if len(seg_x) > 1:
                        # 直接连接数据点，不进行拟合
                        # line = ax.plot(seg_x, seg_y, color=colors[i % len(colors)], linestyle='-', linewidth=1,
                        #              label=f'{int(seg_x[0])}-{int(seg_x[-1])}')
                        
                        # # 如果需要计算APC，可以基于首末两点计算斜率
                        # if len(seg_x) >= 2:
                        #     slope = (seg_y[-1] - seg_y[0]) / (seg_x[-1] - seg_x[0])
                        #     apc = (np.exp(slope) - 1) * 100
                            # 可以选择是否在标签中显示APC
                            # line[0].set_label(f'{int(seg_x[0])}-{int(seg_x[-1])} APC = {apc:.2f}%')
                        
                        # 使用不同颜色绘制线段（直接连接数据点）
                        # line = ax.plot(seg_x, seg_y, color=colors[i % len(colors)], linestyle='-', linewidth=1,
                                    #  label=f'{int(seg_x[0])}-{int(seg_x[-1])} APC = {apc:.2f}*')
                        
                        # 移除中间的APC标签（因为会在图例中显示）
                        # ax.text(mid_x, mid_y, f'APC: {apc:.1f}%', ...) 这行删除
                    
                # 添加图例
                ax.legend(loc='upper right', fontsize=8, bbox_to_anchor=(1.0, 1.0))
                
                # 绘制拐点垂直线
                # for jp in joinpoints:
                    # ax.axvline(jp, color='purple', linestyle='--', alpha=0.3)
                
                # 设置轴和标签
                ax.grid(True, alpha=0.3, linestyle='--')
                ax.set_xlabel('Year')
                ax.set_ylabel(title)
                
                # 设置标题和位置标签
                if row_idx == start_idx and col_idx == 0:
                    ax.text(-0.2, 0.5, location, transform=ax.transAxes, 
                           rotation=90, va='center', ha='right')
                
                # 添加性别标签
                ax.text(-0.1, 0.5, sex, transform=ax.transAxes,
                       rotation=90, va='center', ha='right')
                
                # 设置面板标签
                if row_idx == 0:
                    panel_labels = ['A', 'B', 'C']
                    ax.set_title(f'{panel_labels[col_idx]}\n{title}', loc='left')
                elif row_idx == 3:
                    panel_labels = ['D', 'E', 'F']
                    ax.set_title(f'{panel_labels[col_idx]}', loc='left')
    
                
                # 设置面板标签
                if row_idx == 0:
                    panel_labels = ['A', 'B', 'C']
                    ax.set_title(f'{panel_labels[col_idx]}\n{title}', loc='left')
                elif row_idx == 3:
                    panel_labels = ['D', 'E', 'F']
                    ax.set_title(f'{panel_labels[col_idx]}', loc='left')
    
   
     # 调整布局，增加间距
    plt.tight_layout(pad=3.0)  # 增加整体边距
    
    # # 进一步调整子图间距
    # plt.subplots_adjust(
    #     left=0.08,     # 左边距
    #     right=0.95,    # 右边距
    #     top=0.95,      # 上边距
    #     bottom=0.05,   # 下边距
    #     wspace=0.3,    # 列间距
    #     hspace=0.4     # 行间距
    # )
    
    
    output_path = f'/Users/caijianyu/go/src/code.byted.org/argos/gbd/output/joinpoint_analysis.tif'
    plt.savefig(output_path, dpi=600, bbox_inches='tight', facecolor='white', format='tiff')
    plt.close()

def main():
    # 读取数据
    df = pd.read_csv('/Users/caijianyu/go/src/code.byted.org/argos/gbd/data/raw/breast_cancer.csv')
    pop = pd.read_csv('/Users/caijianyu/go/src/code.byted.org/argos/gbd/data/raw/pop.csv')
    
    # 绘制拐点回归分析图
    plot_joinpoint_analysis(df, df, pop)
    print("拐点回归分析图已保存为TIF格式：/Users/caijianyu/go/src/code.byted.org/argos/gbd/output/joinpoint_analysis.tif (1200 dpi)")

if __name__ == "__main__":
    main()