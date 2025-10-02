import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.interpolate import make_interp_spline
from joinpoint_analysis import find_joinpoints, calculate_slopes
from trend_analysis import calculate_standardized_rate, calculate_asr_mir
from main import process_data  # 添加这行导入

# 定义颜色映射
AGE_COLORS = {
    'All ages': 'black',
    '5-14 years': 'red',
    '15-49 years': 'green',
    '50-69 years': 'cyan',
    '70+ years': 'blue'
}

# 定义年龄组顺序 (用于绘图和图例)
AGE_ORDER = ['15-49 years', '50-69 years', '70+ years']

def predict_future_values(years, values, prediction_end_year=2050):
    """
    基于历史数据预测未来值
    
    Args:
        years: 历史年份数组
        values: 历史值数组
        prediction_end_year: 预测结束年份
    
    Returns:
        预测年份和值的数组
    """
    # 确保数据按年份排序
    sorted_indices = np.argsort(years)
    years = years[sorted_indices]
    values = values[sorted_indices]
    
    # 移除NaN值
    valid_indices = ~np.isnan(values)
    years = years[valid_indices]
    values = values[valid_indices]
    
    if len(years) < 2:
        return np.array([]), np.array([])
    
    # 寻找拐点
    joinpoints = find_joinpoints(years, values, n_joinpoints=2)
    
    # 使用最后一段的斜率进行预测
    last_segment_start = joinpoints[-1] if len(joinpoints) > 0 else years[0]
    last_segment_years = years[years >= last_segment_start]
    last_segment_values = values[years >= last_segment_start]
    
    if len(last_segment_years) < 2:
        # 如果最后一段数据不足，使用所有数据
        last_segment_years = years
        last_segment_values = values
    
    # 线性回归
    slope, intercept = np.polyfit(last_segment_years, last_segment_values, 1)
    
    # 预测未来年份
    future_years = np.arange(years[-1] + 1, prediction_end_year + 1)
    future_values = slope * future_years + intercept
    
    return future_years, future_values

def plot_cancer_predictions(df, pop_df, prediction_start_year=2020):
    """
    绘制中国和G20国家乳腺癌ASIR和ASMR的预测图。
    
    Args:
        df: 包含乳腺癌数据的DataFrame
        pop_df: 包含人口数据的DataFrame
        prediction_start_year: 预测开始的年份
    """
    # 创建2x2的主面板布局
    fig = plt.figure(figsize=(16, 14))
    
    # 定义性别列表
    sexes = ['Both', 'Male', 'Female']
    
    # 创建更紧凑的网格
    gs = fig.add_gridspec(6, 2, hspace=0.05, wspace=0.25)
    
    # 定义四个主面板的位置
    panels = [
        (gs[0:3, 0], 'China', 'Incidence', 'A'),  # 面板A：中国ASIR
        (gs[0:3, 1], 'China', 'Deaths', 'B'),     # 面板B：中国ASMR
        (gs[3:6, 0], 'G20', 'Incidence', 'C'),    # 面板C：G20 ASIR
        (gs[3:6, 1], 'G20', 'Deaths', 'D')        # 面板D：G20 ASMR
    ]
    
    # 获取历史年份范围
    historical_years = df['year'].unique()
    historical_years.sort()
    
    # 为每个面板创建子图
    for panel_idx, (panel_spec, location, measure_type, label) in enumerate(panels):
        # 添加面板标签（只在顶部添加）
        if panel_idx < 2:
            ax = fig.add_subplot(panel_spec)
            ax.set_visible(False)
            ax.text(0.05, 1.05, label, transform=ax.transAxes, fontsize=14, fontweight='bold')
        
        # 筛选当前面板的数据
        location_data = process_data(
            df[
                (df['measure_name'] == measure_type) & 
                (df['location_name'] == location)
            ].copy(),
            pop_df[
                (pop_df['location'] == location)
            ].copy(),
            location
        )
        
        # 为每个性别创建子图
        for sex_idx, sex in enumerate(sexes):
            # 计算子图位置 - 每个面板分成3个子图
            subgs = panel_spec.subgridspec(3, 1, hspace=0.05)
            subax = fig.add_subplot(subgs[sex_idx])
            
            # 筛选性别数据
            sex_data = location_data[location_data['sex_name'] == sex]
            
            # 绘制每个年龄组的线条和计算最大值
            max_rate = 0
            for age_group in AGE_ORDER:
                # 筛选年龄组数据
                age_data = sex_data[sex_data['age_name'] == age_group]
                
                if not age_data.empty:
                    # 计算历史标准化率
                    historical_rates = []
                    historical_years_list = []
                    
                    for year in historical_years:
                        year_data = age_data[age_data['year'] == year]
                        
                        if not year_data.empty:
                            try:
                                # 使用标准化计算方法
                                if measure_type == 'Incidence':
                                    rate = calculate_standardized_rate(year_data, None, 'ASIR')
                                else:  # Deaths
                                    rate = calculate_standardized_rate(year_data, None, 'ASMR')
                                historical_rates.append(rate)
                                historical_years_list.append(year)
                            except Exception as e:
                                print(f"\n计算错误详情:")
                                print(f"地区: {location}, 年份: {year}, 性别: {sex}, 年龄组: {age_group}")
                                print(f"数据形状: {year_data.shape}")
                                print(f"错误信息: {str(e)}")
                                continue
                    
                    if historical_rates:
                        # 转换为numpy数组
                        hist_years = np.array(historical_years_list)
                        hist_rates = np.array(historical_rates)
                        
                        # 预测未来值
                        future_years, future_rates = predict_future_values(hist_years, hist_rates)
                        
                        # 合并历史和预测数据
                        all_years = np.concatenate([hist_years, future_years])
                        all_rates = np.concatenate([hist_rates, future_rates])
                        
                        # 平滑处理
                        if len(all_years) > 3:
                            # 创建更密集的x点用于平滑曲线
                            x_smooth = np.linspace(all_years.min(), all_years.max(), 300)
                            
                            try:
                                # 创建样条模型
                                spl = make_interp_spline(all_years, all_rates, k=3)
                                y_smooth = spl(x_smooth)
                                
                                # 绘制历史数据（实线）
                                hist_mask = x_smooth <= prediction_start_year
                                if np.any(hist_mask):
                                    label_text = age_group if panel_idx == 0 and sex_idx == 0 else ""
                                    subax.plot(x_smooth[hist_mask], y_smooth[hist_mask], 
                                              color=AGE_COLORS.get(age_group, 'gray'),
                                              linewidth=1.5,
                                              label=label_text)
                                
                                # 绘制预测数据（实线）
                                pred_mask = x_smooth >= prediction_start_year
                                if np.any(pred_mask):
                                    subax.plot(x_smooth[pred_mask], y_smooth[pred_mask], 
                                              color=AGE_COLORS.get(age_group, 'gray'),
                                              linewidth=1.5)
                            except:
                                # 如果样条插值失败，则使用原始数据点
                                label_text = age_group if panel_idx == 0 and sex_idx == 0 else ""
                                subax.plot(all_years, all_rates, 
                                          color=AGE_COLORS.get(age_group, 'gray'),
                                          linewidth=1.5,
                                          label=label_text)
                        else:
                            # 数据点太少，使用原始数据
                            label_text = age_group if panel_idx == 0 and sex_idx == 0 else ""
                            subax.plot(all_years, all_rates, 
                                      color=AGE_COLORS.get(age_group, 'gray'),
                                      linewidth=1.5,
                                      label=label_text)
                        
                        # 更新最大值
                        max_rate = max(max_rate, np.nanmax(all_rates))
            
            # 添加垂直虚线表示预测开始的年份
            subax.axvline(x=prediction_start_year, color='gray', linestyle='--', linewidth=1)
            
            # 设置Y轴标签
            if measure_type == 'Incidence':  # ASIR面板
                subax.set_ylabel(f"{sex}\nASIR")
            else:  # ASMR面板
                subax.set_ylabel(f"{sex}\nASMR")
            
            # 设置X轴标签和刻度
            if sex_idx == len(sexes) - 1:  # 最底部的子图
                subax.set_xlabel("Year", fontsize=10)
                subax.tick_params(axis='x', labelsize=8)
            else:
                subax.set_xticklabels([])
            
            # 设置Y轴刻度
            subax.tick_params(axis='y', labelsize=8)
            
            # 添加网格线
            subax.grid(True, linestyle='--', alpha=0.2)
            
            # 在左侧添加位置标签（China/G20）
            if panel_idx in [0, 2] and sex_idx == 1:  # 左侧面板的中间子图
                subax.text(-0.25, 0.5, location, transform=subax.transAxes,
                          rotation=90, va='center', ha='center', fontsize=12)
            
            # 设置Y轴范围 - 根据数据动态调整
            if max_rate > 0:
                # 为了美观，将最大值向上取整并增加一点空间
                y_max = np.ceil(max_rate * 1.1)
                subax.set_ylim(0, y_max)
            else:
                # 如果没有数据，使用默认范围
                if measure_type == 'Incidence':
                    subax.set_ylim(0, 8)
                else:  # Deaths
                    subax.set_ylim(0, 6)
    
    # 创建统一的图例
    handles, labels = [], []
    for age_group in AGE_ORDER:
        handles.append(plt.Line2D([0], [0], color=AGE_COLORS.get(age_group, 'gray'), linewidth=1.5))
        labels.append(age_group)
    
    fig.legend(handles, labels, loc='lower center', ncol=len(AGE_ORDER), 
              bbox_to_anchor=(0.5, 0.01), fontsize=10)
    
    # 添加总标题
    fig.suptitle("Figure 4. Prediction of ASIR and ASMR of breast cancer", fontsize=14, y=0.98)
    
    # 确保output目录存在
    output_dir = "/Users/caijianyu/go/src/code.byted.org/argos/gbd/output"
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存图片
    plt.savefig(f"{output_dir}/figure4_predictions.png", dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # 读取数据
    df = pd.read_csv('/Users/caijianyu/go/src/code.byted.org/argos/gbd/data/raw/breast_cancer.csv')
    pop = pd.read_csv('/Users/caijianyu/go/src/code.byted.org/argos/gbd/data/raw/pop.csv')
    
    # 绘制预测图
    plot_cancer_predictions(df, pop, prediction_start_year=2022)
    print("乳腺癌ASIR和ASMR预测图已保存为：/Users/caijianyu/go/src/code.byted.org/argos/gbd/output/figure4_predictions.png")

if __name__ == "__main__":
    main()