import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats  # 添加这行
from main import (calculate_aapc, calculate_asir, format_number, 
                 create_result_table, filter_data)

def process_g20_countries_data(df_incidence, df_population, measure_name='Incidence'):
    """处理G20各国数据"""
    # 获取所有G20国家列表
    g20_countries = df_population['location'].unique()
    
    # 存储所有国家的结果
    all_countries_results = []
    
    for country in g20_countries:
        # 筛选数据
        country_incidence, country_population = filter_data(df_incidence, df_population, measure_name)
        
        # 处理单个国家数据
        country_grouped = process_country_data(country_incidence, country_population, country)
        country_result_df = create_result_table(country_grouped)
        
        # 添加国家信息
        country_result_df['Country'] = country
        all_countries_results.append(country_result_df)
    
    return all_countries_results

def process_country_data(df_incidence, df_population, country):
    """处理单个国家的数据"""
    # 确保年份和位置的筛选条件
    grouped_incidence = df_incidence[
        (df_incidence['location_name'] == country) &
        (df_incidence['cause_name'] == 'Breast cancer')
    ].groupby(['year', 'sex_name', 'age_name']).agg({
        'val': 'sum',
        'upper': 'sum',
        'lower': 'sum'
    }).reset_index()
    
    # 处理人口数据
    grouped_population = df_population[
        (df_population['location'] == country)
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
    
    return grouped.copy()

def plot_g20_tables(country_results, measure_name='Incidence'):
    """将所有G20国家的数据绘制到一个大表格中"""
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建一个大图
    fig, ax = plt.subplots(figsize=(20, 60))
    ax.axis('tight')
    ax.axis('off')
    
    # 设置表头
    measure_text = 'cases' if measure_name == 'Incidence' else measure_name
    headers = ['Characteristics', 
              f'1990\n{measure_name} {measure_text}\nNo. ×10³ (95% UI)',
              f'1990\nASR per 100000\nNo. (95% UI)',
              f'2021\n{measure_name} {measure_text}\nNo. ×10³ (95% UI)',
              f'2021\nASR per 100000\nNo. (95% UI)',
              '1990-2021\nAAPC\nNo. (95% UI)']
    
    # 准备所有表格数据
    table_data = []
    section_rows = [0]  # 记录需要特殊样式的行
    current_row = 0
    
    for country_df in country_results:
        country_name = country_df['Country'].iloc[0]
        
        # 添加国家标题行
        table_data.append([country_name] + [''] * 5)
        section_rows.append(current_row + 1)
        current_row += 1
        
        for sex in ['Both', 'Male', 'Female']:
            sex_data = country_df[country_df['Sex'] == sex]
            section_rows.append(current_row + 1)  # 性别行需要特殊样式
            for _, row in sex_data.iterrows():
                table_data.append([
                    f"{row['Characteristics']}",
                    row['1990_cases'],
                    row['1990_asir'],
                    row['2021_cases'],
                    row['2021_asir'],
                    row['AAPC']
                ])
                current_row += 1
    
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
    
    # 设置表头样式
    for col in range(6):
        cell = table[0, col]
        cell.set_height(0.15)
        cell.set_text_props(weight='bold')
        cell.set_facecolor('#E6E6E6')
    
    # 设置特殊行的样式
    for row in section_rows:
        for col in range(6):
            cell = table[row, col]
            cell.set_facecolor('#E6E6E6')
            cell.set_text_props(weight='bold')
    
    plt.title(f'Table. The {measure_name.lower()} {measure_text}, age-standardised rates, and temporal trend '
             f'of breast cancer from 1990 to 2021 in G20 countries',
             pad=20, fontsize=12)
    
    plt.savefig(f'/Users/caijianyu/go/src/code.byted.org/argos/gbd/output1/g20_countries_{measure_name.lower()}_table.png',
                dpi=300, bbox_inches='tight')
    plt.close()

# 添加一个新函数来绘制合并的相关性图
def plot_combined_sdi_aapc_correlation(incidence_results, deaths_results, sdi_data):
    """绘制ASIR和ASMR的SDI和AAPC相关性散点图到一个文件中"""
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建图形，包含两个子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 添加国家名称映射
    country_mapping = {
        'Russian Federation': 'Russia',
        'United Kingdom': 'UK',
        'United States of America': 'USA',
    }
    
    # 处理发病率数据
    asir_data = []
    for country_df in incidence_results:
        country_name = country_df['Country'].iloc[0]
        sdi_country = country_mapping.get(country_name, country_name)
        aapc_data = country_df[country_df['Sex'] == 'Both']
        if not aapc_data.empty:
            aapc_value = float(aapc_data['AAPC'].iloc[0].split(' ')[0])
            sdi_match = sdi_data[sdi_data['country'] == sdi_country]
            if not sdi_match.empty:
                sdi_value = sdi_match['1990'].iloc[0]
                population_proxy = float(aapc_data['2021_cases'].iloc[0].split(' ')[0])
                asir_data.append({
                    'country': country_name,
                    'sdi': sdi_value,
                    'aapc': aapc_value,
                    'size': population_proxy
                })
    
    # 处理死亡率数据
    asmr_data = []
    for country_df in deaths_results:
        country_name = country_df['Country'].iloc[0]
        sdi_country = country_mapping.get(country_name, country_name)
        aapc_data = country_df[country_df['Sex'] == 'Both']
        if not aapc_data.empty:
            aapc_value = float(aapc_data['AAPC'].iloc[0].split(' ')[0])
            sdi_match = sdi_data[sdi_data['country'] == sdi_country]
            if not sdi_match.empty:
                sdi_value = sdi_match['1990'].iloc[0]
                population_proxy = float(aapc_data['2021_cases'].iloc[0].split(' ')[0])
                asmr_data.append({
                    'country': country_name,
                    'sdi': sdi_value,
                    'aapc': aapc_value,
                    'size': population_proxy
                })
    
    df_asir = pd.DataFrame(asir_data)
    df_asmr = pd.DataFrame(asmr_data)
    
    # 计算ASIR的相关系数和p值
    correlation_asir = stats.pearsonr(df_asir['sdi'], df_asir['aapc'])[0]
    p_value_asir = stats.pearsonr(df_asir['sdi'], df_asir['aapc'])[1]
    
    # 绘制ASIR散点图 - 使用黑色实心圆点
    # 归一化处理圆圈大小，设置最小和最大尺寸
    min_size = 50
    max_size = 500
    
    # ASIR圆圈大小归一化
    if len(df_asir) > 1:  # 确保有多个数据点才进行归一化
        size_min = df_asir['size'].min()
        size_max = df_asir['size'].max()
        if size_max > size_min:  # 避免除以零
            df_asir['normalized_size'] = min_size + (df_asir['size'] - size_min) * (max_size - min_size) / (size_max - size_min)
        else:
            df_asir['normalized_size'] = min_size
    else:
        df_asir['normalized_size'] = min_size
    
    # ASMR圆圈大小归一化
    if len(df_asmr) > 1:  # 确保有多个数据点才进行归一化
        size_min = df_asmr['size'].min()
        size_max = df_asmr['size'].max()
        if size_max > size_min:  # 避免除以零
            df_asmr['normalized_size'] = min_size + (df_asmr['size'] - size_min) * (max_size - min_size) / (size_max - size_min)
        else:
            df_asmr['normalized_size'] = min_size
    else:
        df_asmr['normalized_size'] = min_size
    
    # 绘制ASIR散点图 - 使用归一化后的大小
    scatter1 = ax1.scatter(df_asir['sdi'], df_asir['aapc'], 
                         s=df_asir['normalized_size'], color='black', edgecolors='black')
    
    # 添加回归线
    z1 = np.polyfit(df_asir['sdi'], df_asir['aapc'], 1)
    p1 = np.poly1d(z1)
    
    # 创建平滑的x值用于绘制回归线和置信区间
    x1_smooth = np.linspace(df_asir['sdi'].min(), df_asir['sdi'].max(), 100)
    y1_pred_smooth = p1(x1_smooth)
    ax1.plot(x1_smooth, y1_pred_smooth, "r-", alpha=0.8)
    
    # 添加国家标签
    for _, row in df_asir.iterrows():
        ax1.annotate(row['country'], 
                   (row['sdi'], row['aapc']),
                   xytext=(5, 5), 
                   textcoords='offset points',
                   fontsize=8)
    
    ax1.set_xlabel('SDI in 1990')
    ax1.set_ylabel('AAPC of ASIR')
    ax1.text(0.05, 0.95, f'r={correlation_asir:.3f}, p={p_value_asir:.3f}',
            transform=ax1.transAxes, fontsize=10)
    
    # 添加ASIR置信区间 - 使用平滑的曲线
    slope1, intercept1, r_value1, p_value1, std_err1 = stats.linregress(df_asir['sdi'], df_asir['aapc'])
    
    # 计算平滑置信区间
    n1 = len(df_asir)
    mean_x1 = np.mean(df_asir['sdi'])
    # 计算每个点的标准误差
    se1 = std_err1 * np.sqrt(1/n1 + (x1_smooth - mean_x1)**2/np.sum((df_asir['sdi'] - mean_x1)**2))
    
    # 绘制平滑的置信区间
    ax1.fill_between(x1_smooth, y1_pred_smooth - 1.96*se1, y1_pred_smooth + 1.96*se1, alpha=0.2, color='gray')
    
    # 计算ASMR的相关系数和p值
    correlation_asmr = stats.pearsonr(df_asmr['sdi'], df_asmr['aapc'])[0]
    p_value_asmr = stats.pearsonr(df_asmr['sdi'], df_asmr['aapc'])[1]
    
    # 绘制ASMR散点图 - 使用黑色实心圆点
    scatter2 = ax2.scatter(df_asmr['sdi'], df_asmr['aapc'], 
                         s=df_asmr['normalized_size'], color='black', edgecolors='black')
    
    # 添加回归线
    z2 = np.polyfit(df_asmr['sdi'], df_asmr['aapc'], 1)
    p2 = np.poly1d(z2)
    
    # 创建平滑的x值用于绘制回归线和置信区间
    x2_smooth = np.linspace(df_asmr['sdi'].min(), df_asmr['sdi'].max(), 100)
    y2_pred_smooth = p2(x2_smooth)
    ax2.plot(x2_smooth, y2_pred_smooth, "r-", alpha=0.8)
    
    # 添加国家标签
    for _, row in df_asmr.iterrows():
        ax2.annotate(row['country'], 
                   (row['sdi'], row['aapc']),
                   xytext=(5, 5), 
                   textcoords='offset points',
                   fontsize=8)
    
    ax2.set_xlabel('SDI in 1990')
    ax2.set_ylabel('AAPC of ASMR')
    ax2.text(0.05, 0.95, f'r={correlation_asmr:.3f}, p={p_value_asmr:.3f}',
            transform=ax2.transAxes, fontsize=10)
    
    # 添加ASMR置信区间 - 使用平滑的曲线
    slope2, intercept2, r_value2, p_value2, std_err2 = stats.linregress(df_asmr['sdi'], df_asmr['aapc'])
    
    # 计算平滑置信区间
    n2 = len(df_asmr)
    mean_x2 = np.mean(df_asmr['sdi'])
    # 计算每个点的标准误差
    se2 = std_err2 * np.sqrt(1/n2 + (x2_smooth - mean_x2)**2/np.sum((df_asmr['sdi'] - mean_x2)**2))
    
    # 绘制平滑的置信区间
    ax2.fill_between(x2_smooth, y2_pred_smooth - 1.96*se2, y2_pred_smooth + 1.96*se2, alpha=0.2, color='gray')
    
    plt.tight_layout()
    plt.savefig('/Users/caijianyu/go/src/code.byted.org/argos/gbd/output1/sdi_aapc_combined_correlation.png',
                dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # 读取数据
    df_data = pd.read_csv('/Users/caijianyu/go/src/code.byted.org/argos/gbd/data/raw/breast_cancel_1.csv')
    df_population = pd.read_csv('/Users/caijianyu/go/src/code.byted.org/argos/gbd/data/raw/pop_g20.csv')
    df_sdi = pd.read_csv('/Users/caijianyu/go/src/code.byted.org/argos/gbd/data/raw/sdi.csv')
    
    # 修改国家名称映射
    country_mapping = {
        'Russian Federation': 'Russia',
        'United Kingdom': 'UK',
        'United States of America': 'USA',
    }
    
    # 获取G20国家列表
    g20_countries = df_population['location'].unique()
    
    # 检查每个国家的匹配情况
    print("\n国家名称匹配检查:")
    for country in g20_countries:
        # 使用映射获取SDI数据中的国家名称
        sdi_country = country_mapping.get(country, country)
        sdi_match = df_sdi[df_sdi['country'] == sdi_country]
        if sdi_match.empty:
            print(f"未匹配: {country} ({sdi_country}) -> 在SDI数据中未找到")
        else:
            print(f"已匹配: {country} ({sdi_country}) -> SDI值: {sdi_match['1990'].iloc[0]}")
    
    # 获取G20国家列表并应用映射
    g20_countries = df_population['location'].unique()
    g20_countries_mapped = [country_mapping.get(country, country) for country in g20_countries]
    
    # 筛选并打印G20国家的SDI数据
    g20_sdi = df_sdi[df_sdi['country'].isin(g20_countries_mapped)]
    print("\nG20国家1990年SDI值:")
    print(g20_sdi[['country', '1990']].sort_values('1990', ascending=False))
    
    # 处理所有指标
    incidence_results = None
    deaths_results = None
    
    for measure_name in ['Incidence', 'Deaths']:  # 添加Deaths指标
        print(f"\n开始处理 {measure_name} 数据...")
        
        # 筛选相应的数据
        df_filtered = df_data[df_data['measure_name'] == measure_name]
        
        # 处理G20各国数据
        country_results = process_g20_countries_data(df_filtered, df_population, measure_name)
        
        # 保存结果以便后续使用
        if measure_name == 'Incidence':
            incidence_results = country_results
        else:
            deaths_results = country_results
        
        # 绘制各国表格
        plot_g20_tables(country_results, measure_name)
        
        print(f"\n已完成 {measure_name} 的数据处理和表格生成")
    
    # 在处理完所有指标后，绘制合并的相关性图
    if incidence_results and deaths_results:
        print("\n开始绘制合并的相关性图...")
        plot_combined_sdi_aapc_correlation(incidence_results, deaths_results, g20_sdi)
        print("合并的相关性图已生成")
        
        # 数据验证
        print(f"\n{measure_name}数据检查:")
        print("数据中的位置名称:", df_filtered['location_name'].unique())
        print("人口数据中的位置名称:", df_population['location'].unique())

if __name__ == "__main__":
    main()