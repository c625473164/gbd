import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_aapc(start_val, end_val, years=29):
    """计算年均变化百分比"""
    try:
        if start_val <= 0 or end_val <= 0:
            return 0.00
        if start_val == end_val:
            return 0.00
        result = ((end_val/start_val)**(1/years) - 1) * 100
        return 0.00 if np.isnan(result) or np.isinf(result) else result
    except Exception as e:
        print(f"计算AAPC时出错: {e}")
        return 0.00

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
    
    # 打印调试信息
    print(f"\n处理 {location} 数据:")
    print(f"发病数据行数: {len(grouped_incidence)}")
    print(f"人口数据行数: {len(grouped_population)}")
    
    # 检查合并前的数据
    print("\n合并前的数据示例:")
    print("发病数据:")
    print(grouped_incidence.head())
    print("\n人口数据:")
    print(grouped_population.head())
    
    # 合并发病数据和人口数据
    grouped = pd.merge(
        grouped_incidence,
        grouped_population,
        on=['year', 'sex_name', 'age_name'],
        how='left'
    )
    
    # 检查合并后的数据
    print("\n合并后的数据示例:")
    print(grouped[['year', 'sex_name', 'age_name', 'val', 'population']].head())
    print(f"合并后的数据行数: {len(grouped)}")
    print(f"包含NaN的行数: {grouped['population'].isna().sum()}")
    
    return grouped.copy()  # 返回副本避免 SettingWithCopyWarning

def calculate_asir(data, year):
    """计算年龄标准化发病率"""
    try:
        year_data = data[data['year'] == year].copy()
        
        # 计算各年龄组的粗发病率
        year_data.loc[:, 'crude_rate'] = (year_data['val'] / year_data['population']) * 100000
        
        # 世界标准人口权重（WHO World Standard）
        age_weights = {
            '5-14 years': 0.17290,
            '15-49 years': 0.52010,
            '50-69 years': 0.16600,
            '70+ years': 0.05275
        }
        
        # 计算ASIR
        total_weight = sum(age_weights.values())
        asir = 0
        
        print("\n各年龄组的ASIR计算过程:")
        for age_group, weight in age_weights.items():
            age_data = year_data[year_data['age_name'] == age_group]
            if not age_data.empty and not age_data['crude_rate'].isna().all():
                crude_rate = age_data['crude_rate'].values[0]
                age_asir = crude_rate * weight
                asir += age_asir
                print(f"{age_group}: crude_rate={crude_rate:.2f}, weight={weight}, contribution={age_asir:.2f}")
            else:
                print(f"{age_group}: 没有数据或数据无效")
        
        result = asir / total_weight
        print(f"\n最终ASIR: {result:.2f}")
        
        return 0.00 if np.isnan(result) else result
        
    except Exception as e:
        print(f"计算ASIR时出错: {e}")
        return 0.00

def format_number(num):
    """将数值转换为 *10³ 格式"""
    if num == 0:
        return "0.00"
    val = num / 1000  # 转换为 10³ 单位
    return f"{val:.2f}×10³"

def create_result_table(grouped):
    # 创建1990年和2021年的数据
    df_1990 = grouped[grouped['year'] == 1990]
    df_2021 = grouped[grouped['year'] == 2021]
    
    results = []
    for sex in ['Both', 'Male', 'Female']:
        sex_data = grouped[grouped['sex_name'] == sex]
        
        # 计算该性别的总和
        total_1990 = df_1990[df_1990['sex_name'] == sex]['val'].sum()
        total_2021 = df_2021[df_2021['sex_name'] == sex]['val'].sum()
        
        # 计算ASIR
        asir_1990 = calculate_asir(sex_data[sex_data['year'] == 1990], 1990)
        asir_2021 = calculate_asir(sex_data[sex_data['year'] == 2021], 2021)
        
        # 添加总计行
        aapc_total = calculate_aapc(total_1990, total_2021)
        results.append({
            'Characteristics': sex,
            'Sex': sex,
            '1990_cases': format_number(total_1990),
            '1990_asir': f"{asir_1990:.2f}",  # 添加ASIR
            '2021_cases': format_number(total_2021),
            '2021_asir': f"{asir_2021:.2f}",  # 添加ASIR
            'AAPC': f"{aapc_total:.2f}"
        })
        
        # 添加各年龄组的数据
        for age_group in ['5-14 years', '15-49 years', '50-69 years', '70+ years']:
            age_data = sex_data[sex_data['age_name'] == age_group]
            if len(age_data) > 0:
                val_1990 = df_1990[(df_1990['sex_name'] == sex) & 
                                  (df_1990['age_name'] == age_group)]['val'].values[0]
                val_2021 = df_2021[(df_2021['sex_name'] == sex) & 
                                  (df_2021['age_name'] == age_group)]['val'].values[0]
                
                # 计算年龄组的ASIR
                age_asir_1990 = calculate_asir(age_data[age_data['year'] == 1990], 1990)
                age_asir_2021 = calculate_asir(age_data[age_data['year'] == 2021], 2021)
                
                aapc = calculate_aapc(val_1990, val_2021)
                
                results.append({
                    'Characteristics': age_group.replace(' years', ''),
                    'Sex': sex,
                    '1990_cases': format_number(val_1990),
                    '1990_asir': f"{age_asir_1990:.2f}",  # 添加ASIR
                    '2021_cases': format_number(val_2021),
                    '2021_asir': f"{age_asir_2021:.2f}",  # 添加ASIR
                    'AAPC': f"{aapc:.2f}"
                })
    
    return pd.DataFrame(results)

def plot_results(result_df):
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建图表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
    
    # 绘制1990年和2021年的对比
    for sex in ['Both', 'Male', 'Female']:
        sex_data = result_df[result_df['Sex'] == sex]
        x = np.arange(len(sex_data))
        width = 0.25
        
        if sex == 'Both':
            offset = -width
        elif sex == 'Male':
            offset = 0
        else:
            offset = width
            
        ax1.bar(x + offset, sex_data['1990_cases'].astype(float), width, label=f'{sex} 1990')
        ax1.bar(x + offset, sex_data['2021_cases'].astype(float), width, 
                bottom=sex_data['1990_cases'].astype(float), label=f'{sex} 2021')
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(sex_data['Characteristics'], rotation=45)
    ax1.set_ylabel('发病数 (Number of Cases)')
    ax1.set_title('1990年和2021年各年龄组发病数对比')
    ax1.legend()
    
    # 绘制AAPC
    sns.barplot(data=result_df, x='Characteristics', y='AAPC', hue='Sex', ax=ax2)
    ax2.set_ylabel('年均变化百分比 (AAPC, %)')
    ax2.set_title('1990-2021年间年均变化百分比')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('/Users/caijianyu/go/src/code.byted.org/argos/gbd/output/incidence_analysis.png', 
                dpi=300, bbox_inches='tight')
    plt.close()

def format_number(num):
    """将数值转换为千分之一"""
    if num == 0:
        return "0.00"
    return f"{(num/1000):.2f}"  # 这里直接除以1000可能导致精度问题

def plot_table(china_df, g20_df):
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, ax = plt.subplots(figsize=(15, 12))
    ax.axis('tight')
    ax.axis('off')
    
    headers = ['Characteristics', 
              '1990\nIncidence cases\nNo. ×10³ (95% UI)',  # 修改表头
              '1990\nASIR per 100000\nNo. (95% UI)',
              '2021\nIncidence cases\nNo. ×10³ (95% UI)',  # 修改表头
              '2021\nASIR per 100000\nNo. (95% UI)',
              '1990-2021\nAAPC\nNo. (95% UI)']
    
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
    
    for idx in section_rows:
        for col in range(6):
            cell = table[idx, col]
            cell.set_facecolor('#E6E6E6')
            cell.set_text_props(weight='bold')
    
    plt.title('Table 1. The incidence cases, age-standardised rates, and temporal trend of breast cancer from 1990 to 2021',
             pad=20, fontsize=12)
    
    plt.savefig('/Users/caijianyu/go/src/code.byted.org/argos/gbd/output/incidence_table.png',
                dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # 读取发病数据
    df_incidence = pd.read_csv('/Users/caijianyu/go/src/code.byted.org/argos/gbd/data/raw/breast_cancer.csv')
    
    # 读取人口数据
    df_population = pd.read_csv('/Users/caijianyu/go/src/code.byted.org/argos/gbd/data/raw/pop.csv')
    
    # 筛选发病数据
    df_incidence = df_incidence[
        (df_incidence['measure_name'] == 'Incidence') & 
        (df_incidence['metric_name'] == 'Number')
    ]
    
    # 筛选人口数据
    df_population = df_population[df_population['measure'] == 'Population']
    
    # 处理China数据
    china_grouped = process_data(df_incidence, df_population, 'China')
    china_result_df = create_result_table(china_grouped)
    
    # 处理G20数据
    g20_grouped = process_data(df_incidence, df_population, 'G20')
    g20_result_df = create_result_table(g20_grouped)
    
    # 绘制合并的表格
    plot_table(china_result_df, g20_result_df)
    
    # 添加数据检查
    print("\n数据检查:")
    print("发病数据中的位置名称:", df_incidence['location_name'].unique())
    print("人口数据中的位置名称:", df_population['location'].unique())
    
    validation_data = df_incidence[
        (df_incidence['location_name'] == 'China') &
        (df_incidence['cause_name'] == 'Thyroid cancer') &
        (df_incidence['year'] == 1990) &
        (df_incidence['sex_name'] == 'Males') &
        (df_incidence['age_name'] == '5-14 years')
    ]
    print("验证原始数据：")
    print(validation_data[['val', 'upper', 'lower']])
    
    print("分析结果已保存为图表：/Users/caijianyu/go/src/code.byted.org/argos/gbd/output/incidence_table.png")
    
    # 在处理数据之前，先检查原始数据的具体值
    print("\n检查数据字段的唯一值：")
    print("Location names:", df_incidence['location_name'].unique())
    print("Sex names:", df_incidence['sex_name'].unique())
    print("Age names:", df_incidence['age_name'].unique())
    print("Years:", df_incidence['year'].unique())
    
    # 检查特定数据
    print("\n检查中国数据：")
    china_data = df_incidence[
        df_incidence['location_name'].str.contains('China', case=False, na=False)
    ]
    if len(china_data) > 0:
        print("找到中国数据样例：")
        print(china_data[['location_name', 'sex_name', 'age_name', 'year', 'val']].head())
    else:
        print("未找到中国相关数据")
    
    print("分析结果已保存为图表：/Users/caijianyu/go/src/code.byted.org/argos/gbd/output/incidence_table.png")

if __name__ == "__main__":
    main()