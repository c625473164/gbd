import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_aapc(start_val, end_val, start_lower=None, start_upper=None, end_lower=None, end_upper=None, years=29):
    """计算年均变化百分比及其区间"""
    try:
        # 计算主值AAPC
        if start_val <= 0 or end_val <= 0:
            return 0.00, 0.00, 0.00
        if start_val == end_val:
            return 0.00, 0.00, 0.00
        
        result = ((end_val/start_val)**(1/years) - 1) * 100
        main_aapc = 0.00 if np.isnan(result) or np.isinf(result) else result
        
        # 计算下限AAPC（使用start_upper和end_lower计算最小变化率）
        lower_aapc = 0.00
        if start_upper is not None and end_lower is not None and start_upper > 0 and end_lower > 0:
            lower_result = ((end_lower/start_upper)**(1/years) - 1) * 100
            lower_aapc = 0.00 if np.isnan(lower_result) or np.isinf(lower_result) else lower_result
        
        # 计算上限AAPC（使用start_lower和end_upper计算最大变化率）
        upper_aapc = 0.00
        if start_lower is not None and end_upper is not None and start_lower > 0 and end_upper > 0:
            upper_result = ((end_upper/start_lower)**(1/years) - 1) * 100
            upper_aapc = 0.00 if np.isnan(upper_result) or np.isinf(upper_result) else upper_result
        
        return main_aapc, lower_aapc, upper_aapc
    except Exception as e:
        print(f"计算AAPC时出错: {e}")
        return 0.00, 0.00, 0.00

def process_data(df_incidence, df_population, location):
    # 处理发病数据
    if location == 'China':
        location_filter = "China"
    elif location == 'G20':
        location_filter = 'G20'
    else:
        location_filter = location
    
    # 确保年份和位置的筛选条件
    grouped_incidence = df_incidence[
        (df_incidence['location_name'] == location_filter) &
        (df_incidence['cause_name'] == 'Breast cancer')
    ].groupby(['year', 'sex_name', 'age_name']).agg({
        'val': 'sum',
        'upper': 'sum',
        'lower': 'sum'
    }).reset_index()
    
    # 处理人口数据，确保正确匹配location
    grouped_population = df_population[
        (df_population['location'] == location_filter)
    ].groupby(['year', 'sex', 'age']).agg({
        'val': 'sum'
    }).reset_index()
    
    # 重命名列以匹配
    grouped_population = grouped_population.rename(columns={
        'sex': 'sex_name',
        'age': 'age_name',
        'val': 'population'
    })
    # 合并发病数据和人口数据
    grouped = pd.merge(
        grouped_incidence,
        grouped_population,
        on=['year', 'sex_name', 'age_name'],
        how='left'
    )
    
    return grouped.copy()  # 返回副本避免 SettingWithCopyWarning

def calculate_asir(data, year, value_type='val'):
    try:
        year_data = data[data['year'] == year].copy()
        
        # 计算各年龄组的粗发病率
        if value_type == 'val':
            year_data.loc[:, 'crude_rate'] = (year_data['val'] / year_data['population']) * 100000
        elif value_type == 'upper':
            year_data.loc[:, 'crude_rate'] = (year_data['upper'] / year_data['population']) * 100000
        elif value_type == 'lower':
            year_data.loc[:, 'crude_rate'] = (year_data['lower'] / year_data['population']) * 100000
        
        # 世界标准人口权重（WHO World Standard）
        age_weights = {
            '5-14 years': 0.17290,
            '15-49 years': 0.52010,
            '50-69 years': 0.16600,
            '70+ years': 0.05275
        }
        
        # 计算ASIR
        asir = 0
        total_weight = sum(age_weights.values())
        for age_group, weight in age_weights.items():
            age_data = year_data[year_data['age_name'] == age_group]
            if not age_data.empty and not age_data['crude_rate'].isna().all():
                crude_rate = age_data['crude_rate'].values[0]
                age_asir = crude_rate * weight
                asir += age_asir
        result = asir / total_weight

        return 0.00 if np.isnan(result) else result
        
    except Exception as e:
        print(f"计算ASIR时出错: {e}")
        return 0.00

def format_number(val, upper=None, lower=None):
    """将数值转换为千分之一，并添加区间"""
    if val == 0:
        return "0.00"
    
    # 转换主值
    formatted_val = f"{(val/1000):.2f}"
    
    # 如果提供了上下限，则添加区间
    if upper is not None and lower is not None:
        formatted_upper = f"{(upper/1000):.2f}"
        formatted_lower = f"{(lower/1000):.2f}"
        return f"{formatted_val} ({formatted_lower}, {formatted_upper})"
    
    return formatted_val

def create_result_table(grouped):
    # 创建1990年和2021年的数据
    df_1990 = grouped[grouped['year'] == 1990]
    df_2021 = grouped[grouped['year'] == 2021]
    
    results = []
    for sex in ['Both', 'Male', 'Female']:
        sex_data = grouped[grouped['sex_name'] == sex]
        
        # 计算该性别的总和
        total_1990 = df_1990[df_1990['sex_name'] == sex]['val'].sum()
        total_1990_upper = df_1990[df_1990['sex_name'] == sex]['upper'].sum()
        total_1990_lower = df_1990[df_1990['sex_name'] == sex]['lower'].sum()
        
        total_2021 = df_2021[df_2021['sex_name'] == sex]['val'].sum()
        total_2021_upper = df_2021[df_2021['sex_name'] == sex]['upper'].sum()
        total_2021_lower = df_2021[df_2021['sex_name'] == sex]['lower'].sum()
        
        # 在create_result_table函数中修改ASIR计算部分
        
        # 计算ASIR及其区间
        asir_1990 = calculate_asir(sex_data[sex_data['year'] == 1990], 1990, 'val')
        asir_1990_upper = calculate_asir(sex_data[sex_data['year'] == 1990], 1990, 'upper')
        asir_1990_lower = calculate_asir(sex_data[sex_data['year'] == 1990], 1990, 'lower')
        
        asir_2021 = calculate_asir(sex_data[sex_data['year'] == 2021], 2021, 'val')
        asir_2021_upper = calculate_asir(sex_data[sex_data['year'] == 2021], 2021, 'upper')
        asir_2021_lower = calculate_asir(sex_data[sex_data['year'] == 2021], 2021, 'lower')
        
        # 格式化ASIR值，添加区间
        asir_1990_formatted = f"{asir_1990:.2f} ({asir_1990_lower:.2f}, {asir_1990_upper:.2f})"
        asir_2021_formatted = f"{asir_2021:.2f} ({asir_2021_lower:.2f}, {asir_2021_upper:.2f})"
        
        # 在results.append中使用格式化后的ASIR值
        # 添加总计行
        aapc_total, aapc_lower, aapc_upper = calculate_aapc(
        total_1990, total_2021,
        total_1990_lower, total_1990_upper,
        total_2021_lower, total_2021_upper
        )
        aapc_formatted = f"{aapc_total:.2f} ({aapc_lower:.2f}, {aapc_upper:.2f})"
        
        results.append({
        'Characteristics': sex,
        'Sex': sex,
        '1990_cases': format_number(total_1990, total_1990_upper, total_1990_lower),
        '1990_asir': asir_1990_formatted,  # 使用带区间的ASIR
        '2021_cases': format_number(total_2021, total_2021_upper, total_2021_lower),
        '2021_asir': asir_2021_formatted,  # 使用带区间的ASIR
        'AAPC': aapc_formatted
        })
        
        # 添加各年龄组的数据
        for age_group in ['5-14 years', '15-49 years', '50-69 years', '70+ years']:
            age_data = sex_data[sex_data['age_name'] == age_group]
            if len(age_data) > 0:
                # 获取1990年数据
                age_1990 = df_1990[(df_1990['sex_name'] == sex) & 
                                  (df_1990['age_name'] == age_group)]
                val_1990 = age_1990['val'].values[0] if len(age_1990) > 0 else 0
                upper_1990 = age_1990['upper'].values[0] if len(age_1990) > 0 else 0
                lower_1990 = age_1990['lower'].values[0] if len(age_1990) > 0 else 0
                
                # 获取2021年数据
                age_2021 = df_2021[(df_2021['sex_name'] == sex) & 
                                  (df_2021['age_name'] == age_group)]
                val_2021 = age_2021['val'].values[0] if len(age_2021) > 0 else 0
                upper_2021 = age_2021['upper'].values[0] if len(age_2021) > 0 else 0
                lower_2021 = age_2021['lower'].values[0] if len(age_2021) > 0 else 0
                
                # 计算年龄组的ASIR
                # 计算年龄组的ASIR及其区间
                age_asir_1990 = calculate_asir(age_data[age_data['year'] == 1990], 1990, 'val')
                age_asir_1990_upper = calculate_asir(age_data[age_data['year'] == 1990], 1990, 'upper')
                age_asir_1990_lower = calculate_asir(age_data[age_data['year'] == 1990], 1990, 'lower')
                
                age_asir_2021 = calculate_asir(age_data[age_data['year'] == 2021], 2021, 'val')
                age_asir_2021_upper = calculate_asir(age_data[age_data['year'] == 2021], 2021, 'upper')
                age_asir_2021_lower = calculate_asir(age_data[age_data['year'] == 2021], 2021, 'lower')
                
                # 格式化年龄组ASIR值，添加区间
                age_asir_1990_formatted = f"{age_asir_1990:.2f} ({age_asir_1990_lower:.2f}, {age_asir_1990_upper:.2f})"
                age_asir_2021_formatted = f"{age_asir_2021:.2f} ({age_asir_2021_lower:.2f}, {age_asir_2021_upper:.2f})"
                
                # 在results.append中使用格式化后的年龄组ASIR值
                # 计算年龄组的AAPC及其区间
                aapc, aapc_lower, aapc_upper = calculate_aapc(
                val_1990, val_2021,
                lower_1990, upper_1990,
                lower_2021, upper_2021
                )
                aapc_formatted = f"{aapc:.2f} ({aapc_lower:.2f}, {aapc_upper:.2f})"
                
                results.append({
                    'Characteristics': age_group.replace(' years', ''),
                    'Sex': sex,
                    '1990_cases': format_number(val_1990, upper_1990, lower_1990),
                    '1990_asir': age_asir_1990_formatted,  # 使用带区间的ASIR
                    '2021_cases': format_number(val_2021, upper_2021, lower_2021),
                    '2021_asir': age_asir_2021_formatted,  # 使用带区间的ASIR
                    'AAPC': aapc_formatted
                })
    
    return pd.DataFrame(results)


def plot_table(china_df, g20_df, measure_name='Incidence', excel_output_path=None):
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, ax = plt.subplots(figsize=(15, 12))
    ax.axis('tight')
    ax.axis('off')

    header = "ASIR"
    if measure_name == 'Deaths':
        header = "ASMR"
    elif measure_name == 'DALYs':
        header = "ASDR"
    
    # 根据 measure_name 动态设置表头
    measure_text = 'cases' if measure_name == 'Incidence' else measure_name
    headers = ['Characteristics', 
              f'1990\n{measure_name} {measure_text}\nNo. ×10³ (95% CI)',
              f'1990\n{header} per 100000\nNo. (95% CI)',
              f'2021\n{measure_name} {measure_text}\nNo. ×10³ (95% CI)',
              f'2021\n{header} per 100000\nNo. (95% CI)',
              '1990-2021\nAAPC\nNo. (95% CI)']
    
    table_data = []
    # 添加China数据
    table_data.append(['China*'] + [''] * 5)
    for sex in ['Both', 'Male', 'Female']:
        sex_data = china_df[china_df['Sex'] == sex]
        for _, row in sex_data.iterrows():
            table_data.append([
                f"{row['Characteristics']}",
                row['1990_cases'],
                row['1990_asir'],
                row['2021_cases'],
                row['2021_asir'],
                row['AAPC']
            ])
    
    # 添加G20数据
    table_data.append(['G20'] + [''] * 5)  # 添加G20标题行
    for sex in ['Both', 'Male', 'Female']:
        sex_data = g20_df[g20_df['Sex'] == sex]
        for _, row in sex_data.iterrows():
            table_data.append([
                f"{row['Characteristics']}",
                row['1990_cases'],
                row['1990_asir'],
                row['2021_cases'],
                row['2021_asir'],
                row['AAPC']
            ])
    
    # 创建表格
    table = ax.table(cellText=table_data,
                    colLabels=headers,
                    loc='center',
                    cellLoc='center',
                    colWidths=[0.15, 0.17, 0.17, 0.17, 0.17, 0.17])
    
    # 设置表格样式
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    
    # 为所有标题行添加样式
    header_rows = [0]  # 表头
    section_rows = [1]  # China*
    for i, row in enumerate(table_data):
        if row[0] in ['Both', 'Male', 'Female', 'G20']:
            section_rows.append(i + 1)  # +1 是因为表头占用了第一行
    
    # 设置样式
    for idx in header_rows:
        for col in range(6):
            cell = table[idx, col]
            cell.set_height(0.15)  # 表头高度
            cell.set_text_props(weight='bold')
            cell.set_facecolor('#E6E6E6')
            cell.set_text_props(weight='bold')
    
    # 为China*和G20标题行设置特殊背景色
    china_row = 1  # China*行在第1行（表头是第0行）
    g20_row = None
    
    # 找到G20标题行的位置
    for i, row in enumerate(table_data):
        if row[0] == 'G20':
            g20_row = i + 1  # +1 因为表头占第0行
            break
    
    # 为China*标题行设置浅灰色背景
    for col in range(6):
        cell = table[china_row, col]
        cell.set_text_props(weight='bold')
        cell.set_facecolor('#D3D3D3')  # 浅灰色背景
    
    # 为G20标题行设置浅灰色背景
    if g20_row is not None:
        for col in range(6):
            cell = table[g20_row, col]
            cell.set_text_props(weight='bold')
            cell.set_facecolor('#D3D3D3')  # 浅灰色背景

    plt.title(f'The {measure_name.lower()} {measure_text}, age-standardised rates, and temporal trend of breast cancer from 1990 to 2021',
             pad=20, fontsize=12)
    
    plt.savefig(f'/Users/caijianyu/go/src/code.byted.org/argos/gbd/output/{measure_name.lower()}_table.tif',
                dpi=600, bbox_inches='tight', facecolor='white', format='tiff')
    plt.close()

    # 输出到Excel
    if excel_output_path:
        # 创建一个包含表头的DataFrame
        excel_df = pd.DataFrame(table_data, columns=headers)
        excel_df.to_excel(excel_output_path, index=False)
        print(f"表格数据已保存到: {excel_output_path}")

def plot_trend_lines(grouped_data, location_name, measure_name='Incidence'):
    """
    绘制1990-2021年间的ASIR/ASMR/ASDR趋势线图
    """
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    
    plt.figure(figsize=(14, 8))
    
    # 计算每年的年龄标准化率
    years = range(1990, 2022)
    trends_data = []
    
    for year in years:
        for sex in ['Both', 'Male', 'Female']:
            sex_data = grouped_data[grouped_data['sex_name'] == sex]
            year_data = sex_data[sex_data['year'] == year]
            
            if not year_data.empty:
                # 计算年龄标准化率
                asir = calculate_asir(year_data, year, 'val')
                asir_upper = calculate_asir(year_data, year, 'upper')
                asir_lower = calculate_asir(year_data, year, 'lower')
                
                trends_data.append({
                    'year': year,
                    'sex': sex,
                    'asir': asir,
                    'asir_upper': asir_upper,
                    'asir_lower': asir_lower
                })
    
    trends_df = pd.DataFrame(trends_data)
    
    # 设置颜色和样式
    colors = {'Both': '#2E86AB', 'Male': '#A23B72', 'Female': '#F18F01'}
    linestyles = {'Both': '-', 'Male': '--', 'Female': '-.'}
    
    # 绘制趋势线
    for sex in ['Both', 'Male', 'Female']:
        sex_data = trends_df[trends_df['sex'] == sex]
        
        # 主线
        plt.plot(sex_data['year'], sex_data['asir'], 
                color=colors[sex], linestyle=linestyles[sex],
                linewidth=2.5, label=sex, marker='o', markersize=3)
        
        # 置信区间（可选，如果数据质量好的话）
        plt.fill_between(sex_data['year'], 
                        sex_data['asir_lower'], 
                        sex_data['asir_upper'],
                        color=colors[sex], alpha=0.2)
    
    # 设置图表样式
    plt.xlabel('Year', fontsize=14, fontweight='bold')
    
    # 根据measure_name设置y轴标签
    if measure_name == 'Incidence':
        ylabel = 'ASIR (per 100,000)'
        title_suffix = 'Incidence'
    elif measure_name == 'Deaths':
        ylabel = 'ASMR (per 100,000)'
        title_suffix = 'Mortality'
    elif measure_name == 'DALYs':
        ylabel = 'ASDR (per 100,000)'
        title_suffix = 'DALYs'
    else:
        ylabel = 'Age-standardized Rate (per 100,000)'
        title_suffix = measure_name
    
    plt.ylabel(ylabel, fontsize=14, fontweight='bold')
    plt.title(f'{location_name} Breast Cancer {title_suffix} Trends (1990-2021)', 
             fontsize=16, fontweight='bold', pad=20)
    
    # 设置图例 - 放在图表外部右侧
    plt.legend(frameon=True, fancybox=True, shadow=True, 
              bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
    
    # 设置网格
    plt.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # 设置坐标轴
    plt.xlim(1989, 2022)
    plt.xticks(range(1990, 2022, 5), fontsize=11)
    plt.yticks(fontsize=11)
    
    # 添加趋势分析文本
    # 计算总体趋势
    both_data = trends_df[trends_df['sex'] == 'Both']
    start_rate = both_data[both_data['year'] == 1990]['asir'].values[0]
    end_rate = both_data[both_data['year'] == 2021]['asir'].values[0]
    aapc_trend, _, _ = calculate_aapc(start_rate, end_rate, years=31)
    
    trend_text = f'Overall Trend (Both): AAPC = {aapc_trend:.2f}%'
    plt.text(0.02, 0.98, trend_text, transform=plt.gca().transAxes, 
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # 保存图片 - 高分辨率TIF格式
    output_path = f'/Users/caijianyu/go/src/code.byted.org/argos/gbd/output/China_vs_G20_{measure_name.lower()}_trends_comparison.tif'
    plt.savefig(output_path, dpi=600, bbox_inches='tight', facecolor='white', format='tiff')
    plt.close()
    
    print(f"Combined trend chart saved to: {output_path}")
    # plt.show()
    
    print(f"Trend chart saved to: {output_path}")
    return trends_df

def plot_combined_trend_lines(china_grouped, g20_grouped, measure_name='Incidence'):
    """
    绘制中国和G20的趋势线图合并为一个包含两个子图的大图
    """
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建包含两个子图的图表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # 计算每年的年龄标准化率的函数
    def calculate_trends_data(grouped_data):
        years = range(1990, 2022)
        trends_data = []
        
        for year in years:
            for sex in ['Both', 'Male', 'Female']:
                sex_data = grouped_data[grouped_data['sex_name'] == sex]
                year_data = sex_data[sex_data['year'] == year]
                
                if not year_data.empty:
                    # 计算年龄标准化率
                    asir = calculate_asir(year_data, year, 'val')
                    asir_upper = calculate_asir(year_data, year, 'upper')
                    asir_lower = calculate_asir(year_data, year, 'lower')
                    
                    trends_data.append({
                        'year': year,
                        'sex': sex,
                        'asir': asir,
                        'asir_upper': asir_upper,
                        'asir_lower': asir_lower
                    })
        
        return pd.DataFrame(trends_data)
    
    # 计算中国和G20的趋势数据
    china_trends_df = calculate_trends_data(china_grouped)
    g20_trends_df = calculate_trends_data(g20_grouped)
    
    # 设置颜色和样式
    colors = {'Both': '#2E86AB', 'Male': '#A23B72', 'Female': '#F18F01'}
    linestyles = {'Both': '-', 'Male': '--', 'Female': '-.'}
    
    # 绘制中国趋势图（左子图）
    for sex in ['Both', 'Male', 'Female']:
        sex_data = china_trends_df[china_trends_df['sex'] == sex]
        
        # 主线
        ax1.plot(sex_data['year'], sex_data['asir'], 
                color=colors[sex], linestyle=linestyles[sex],
                linewidth=2.5, label=sex, marker='o', markersize=3)
        
        # 置信区间
        ax1.fill_between(sex_data['year'], 
                        sex_data['asir_lower'], 
                        sex_data['asir_upper'],
                        color=colors[sex], alpha=0.2)
    
    # 绘制G20趋势图（右子图）
    for sex in ['Both', 'Male', 'Female']:
        sex_data = g20_trends_df[g20_trends_df['sex'] == sex]
        
        # 主线
        ax2.plot(sex_data['year'], sex_data['asir'], 
                color=colors[sex], linestyle=linestyles[sex],
                linewidth=2.5, label=sex, marker='o', markersize=3)
        
        # 置信区间
        ax2.fill_between(sex_data['year'], 
                        sex_data['asir_lower'], 
                        sex_data['asir_upper'],
                        color=colors[sex], alpha=0.2)
    
    # 根据measure_name设置y轴标签
    if measure_name == 'Incidence':
        ylabel = 'ASIR (per 100,000)'
        title_suffix = 'Incidence'
    elif measure_name == 'Deaths':
        ylabel = 'ASMR (per 100,000)'
        title_suffix = 'Mortality'
    elif measure_name == 'DALYs':
        ylabel = 'ASDR (per 100,000)'
        title_suffix = 'DALYs'
    else:
        ylabel = 'Age-standardized Rate (per 100,000)'
        title_suffix = measure_name
    
    # 设置左子图（中国）
    ax1.set_xlabel('Year', fontsize=14, fontweight='bold')
    ax1.set_ylabel(ylabel, fontsize=14, fontweight='bold')
    ax1.set_title(f'China Breast Cancer {title_suffix} Trends (1990-2021)', 
                 fontsize=16, fontweight='bold', pad=20)
    ax1.legend(frameon=True, fancybox=True, shadow=True, 
              bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
    ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax1.set_xlim(1989, 2022)
    ax1.set_xticks(range(1990, 2022, 5))
    ax1.tick_params(axis='both', labelsize=11)
    
    # 设置右子图（G20）
    ax2.set_xlabel('Year', fontsize=14, fontweight='bold')
    ax2.set_ylabel(ylabel, fontsize=14, fontweight='bold')
    ax2.set_title(f'G20 Breast Cancer {title_suffix} Trends (1990-2021)', 
                 fontsize=16, fontweight='bold', pad=20)
    ax2.legend(frameon=True, fancybox=True, shadow=True, 
              bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
    ax2.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax2.set_xlim(1989, 2022)
    ax2.set_xticks(range(1990, 2022, 5))
    ax2.tick_params(axis='both', labelsize=11)
    
    # 添加趋势分析文本
    # 中国趋势
    china_both_data = china_trends_df[china_trends_df['sex'] == 'Both']
    china_start_rate = china_both_data[china_both_data['year'] == 1990]['asir'].values[0]
    china_end_rate = china_both_data[china_both_data['year'] == 2021]['asir'].values[0]
    china_aapc_trend, _, _ = calculate_aapc(china_start_rate, china_end_rate, years=31)
    
    china_trend_text = f'Overall Trend (Both): AAPC = {china_aapc_trend:.2f}%'
    ax1.text(0.02, 0.98, china_trend_text, transform=ax1.transAxes, 
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # G20趋势
    g20_both_data = g20_trends_df[g20_trends_df['sex'] == 'Both']
    g20_start_rate = g20_both_data[g20_both_data['year'] == 1990]['asir'].values[0]
    g20_end_rate = g20_both_data[g20_both_data['year'] == 2021]['asir'].values[0]
    g20_aapc_trend, _, _ = calculate_aapc(g20_start_rate, g20_end_rate, years=31)
    
    g20_trend_text = f'Overall Trend (Both): AAPC = {g20_aapc_trend:.2f}%'
    ax2.text(0.02, 0.98, g20_trend_text, transform=ax2.transAxes, 
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout(pad=3.0, w_pad=5.0)
    
    # 保存图片 - 高分辨率TIF格式
    output_path = f'/Users/caijianyu/go/src/code.byted.org/argos/gbd/output/China_vs_G20_{measure_name.lower()}_trends_comparison.tif'
    plt.savefig(output_path, dpi=600, bbox_inches='tight', facecolor='white', format='tiff')
    plt.close()
    
    print(f"Combined trend chart saved to: {output_path}")
    return china_trends_df, g20_trends_df

def filter_data(df_incidence, df_population, measure_name='Incidence'):
    """筛选发病/死亡数据和人口数据"""
    # 筛选发病/死亡数据
    df_incidence = df_incidence[
        (df_incidence['measure_name'] == measure_name) & 
        (df_incidence['metric_name'] == 'Number')
    ]
    
    # 筛选人口数据
    df_population = df_population[df_population['measure'] == 'Population']
    
    return df_incidence, df_population

def process_and_validate_data(df_incidence, df_population, measure_name='Incidence'):
    """处理和验证数据"""
    # 筛选数据
    df_incidence, df_population = filter_data(df_incidence, df_population, measure_name)
    
    # 处理China数据
    china_grouped = process_data(df_incidence, df_population, 'China')
    china_result_df = create_result_table(china_grouped)
    
    # 处理G20数据
    g20_grouped = process_data(df_incidence, df_population, 'G20')
    g20_result_df = create_result_table(g20_grouped)
    
    # 定义Excel输出路径
    excel_file_path = f'/Users/caijianyu/go/src/code.byted.org/argos/gbd/output/{measure_name.lower()}_table.xlsx'
    
    # 绘制合并的表格并输出Excel
    plot_table(china_result_df, g20_result_df, measure_name, excel_output_path=excel_file_path)
    
    # 新增：绘制趋势线图
    print(f"\n正在生成{measure_name}趋势线图...")
    china_trends = plot_trend_lines(china_grouped, 'China', measure_name)
    g20_trends = plot_trend_lines(g20_grouped, 'G20', measure_name)
    
    # 新增：绘制合并的趋势线图
    print(f"\n正在生成{measure_name}合并趋势线图...")
    china_trends_combined, g20_trends_combined = plot_combined_trend_lines(china_grouped, g20_grouped, measure_name)
    
    # 添加数据检查
    print("\n数据检查:")
    print("发病数据中的位置名称:", df_incidence['location_name'].unique())
    print("人口数据中的位置名称:", df_population['location'].unique())
    
    validation_data = df_incidence[
        (df_incidence['location_name'] == 'China') &
        (df_incidence['cause_name'] == 'Breast cancer') &
        (df_incidence['year'] == 1990) &
        (df_incidence['sex_name'] == 'Both') &
        (df_incidence['age_name'] == '15-49 years') 
    ]
    print("验证原始数据：")
    print(validation_data[['measure_name', 'metric_name','cause_name','sex_name', 'val', 'upper', 'lower']])
    
    print(f"分析结果已保存为图表：/Users/caijianyu/go/src/code.byted.org/argos/gbd/output/{measure_name.lower()}_table.tif")
    
    return china_trends, g20_trends

def main():
    # 读取发病数据
    df_incidence = pd.read_csv('/Users/caijianyu/go/src/code.byted.org/argos/gbd/data/raw/breast_cancer.csv')
    
    # 读取人口数据
    df_population = pd.read_csv('/Users/caijianyu/go/src/code.byted.org/argos/gbd/data/raw/pop.csv')
    
    # # 处理和验证数据
    process_and_validate_data(df_incidence, df_population, 'Incidence')

    # 处理和验证数据
    process_and_validate_data(df_incidence, df_population, 'Deaths')

    # 处理和验证数据
    process_and_validate_data(df_incidence, df_population, 'DALYs')
if __name__ == "__main__":
    main()