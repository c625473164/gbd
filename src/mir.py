import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def calculate_mir(deaths_data, incidence_data, population_data, location, year):
    """计算特定年份和地区的MIR，包括区间值"""
    # 处理死亡和发病数据
    deaths = deaths_data[
        (deaths_data['location_name'] == location) &
        (deaths_data['year'] == year) &
        (deaths_data['cause_name'] == 'Breast cancer')
    ]
    
    
    incidence = incidence_data[
        (incidence_data['location_name'] == location) &
        (incidence_data['year'] == year) &
        (incidence_data['cause_name'] == 'Breast cancer')
    ]
  
    results = []
    for sex in ['Both', 'Male', 'Female']:
        sex_deaths = deaths[deaths['sex_name'] == sex]
        sex_incidence = incidence[incidence['sex_name'] == sex]
        
        # 计算总体MIR及其区间
        total_deaths_val = sex_deaths['val'].sum()
        total_deaths_upper = sex_deaths['upper'].sum()
        total_deaths_lower = sex_deaths['lower'].sum()
        
        total_incidence_val = sex_incidence['val'].sum()
        total_incidence_upper = sex_incidence['upper'].sum()
        total_incidence_lower = sex_incidence['lower'].sum()
        
        mir_val = total_deaths_val / total_incidence_val if total_incidence_val > 0 else 0
        # 计算区间 - 使用上限死亡除以下限发病得到上限MIR，下限死亡除以上限发病得到下限MIR
        mir_upper = total_deaths_upper / total_incidence_lower if total_incidence_lower > 0 else 0
        mir_lower = total_deaths_lower / total_incidence_upper if total_incidence_upper > 0 else 0
        
        results.append({
            'Characteristics': sex,
            'Sex': sex,
            'MIR': round(mir_val, 2),
            'MIR_upper': round(mir_upper, 2),
            'MIR_lower': round(mir_lower, 2)
        })
        
        # 计算各年龄组的MIR及其区间
        for age_group in ['5-14 years', '15-49 years', '50-69 years', '70+ years']:
            age_deaths_val = sex_deaths[sex_deaths['age_name'] == age_group]['val'].sum()
            age_deaths_upper = sex_deaths[sex_deaths['age_name'] == age_group]['upper'].sum()
            age_deaths_lower = sex_deaths[sex_deaths['age_name'] == age_group]['lower'].sum()
            
            age_incidence_val = sex_incidence[sex_incidence['age_name'] == age_group]['val'].sum()
            age_incidence_upper = sex_incidence[sex_incidence['age_name'] == age_group]['upper'].sum()
            age_incidence_lower = sex_incidence[sex_incidence['age_name'] == age_group]['lower'].sum()
            
            age_mir_val = age_deaths_val / age_incidence_val if age_incidence_val > 0 else 0
            age_mir_upper = age_deaths_upper / age_incidence_lower if age_incidence_lower > 0 else 0
            age_mir_lower = age_deaths_lower / age_incidence_upper if age_incidence_upper > 0 else 0
            
            results.append({
                'Characteristics': age_group.replace(' years', ''),
                'Sex': sex,
                'MIR': round(age_mir_val, 2),
                'MIR_upper': round(age_mir_upper, 2),
                'MIR_lower': round(age_mir_lower, 2)
            })
    
    return pd.DataFrame(results)

def plot_mir_table(china_df, g20_df):
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 增加图形的整体高度
    fig, ax = plt.subplots(figsize=(12, 16))
    ax.axis('tight')
    ax.axis('off')
    
    headers = ['Characteristics', 
              '1990\nMIR',
              '2021\nMIR',
              '1990-2021\nAAPC\nNo. (95% CI)']
    
    table_data = []
    # 添加China数据
    table_data.append(['China*'] + [''] * 3)
    for sex in ['Both', 'Male', 'Female']:
        sex_data = china_df[china_df['Sex'] == sex]
        for _, row in sex_data.iterrows():
            table_data.append([
                f"{row['Characteristics']}",
                f"{row['MIR_1990']:.2f}",  # 修改列名
                f"{row['MIR_2021']:.2f}",  # 修改列名
                f"{row['AAPC']}"
            ])
    
    # 添加G20数据
    table_data.append(['G20'] + [''] * 3)
    for sex in ['Both', 'Male', 'Female']:
        sex_data = g20_df[g20_df['Sex'] == sex]
        for _, row in sex_data.iterrows():
            table_data.append([
                f"{row['Characteristics']}",
                f"{row['MIR_1990']:.2f}",  # 修改列名
                f"{row['MIR_2021']:.2f}",  # 修改列名
                f"{row['AAPC']}"
            ])
    
    table = ax.table(cellText=table_data,
                    colLabels=headers,
                    loc='center',
                    cellLoc='center',
                    colWidths=[0.25, 0.25, 0.25, 0.25])
    
    # 设置表格样式
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    
    # 增加表格的缩放比例，特别是高度
    table.scale(1.2, 1.8)  # 将高度缩放从1.5增加到1.8
    
    # 调整表头行的高度
    for col in range(len(headers)):
        cell = table[0, col]
        cell.set_height(0.15)  # 增加表头单元格的高度
        cell.set_facecolor('#E6E6E6')
        cell.set_text_props(weight='bold')
    
    # 设置标题行和分组行的样式
    header_rows = [0]
    section_rows = [1]  # China*
    for i, row in enumerate(table_data):
        if row[0] in ['Both', 'Male', 'Female', 'G20']:
            section_rows.append(i + 1)
    
    for idx in header_rows + section_rows:
        for col in range(len(headers)):
            cell = table[idx, col]
            cell.set_facecolor('#E6E6E6')
            cell.set_text_props(weight='bold')
    
    plt.title('The MIR and temporal trend of breast cancer from 1990 to 2021',  # 修改年份
             pad=20, fontsize=12)
    
    plt.savefig('/Users/caijianyu/go/src/code.byted.org/argos/gbd/output/mir_table.tif',
                dpi=600, bbox_inches='tight', facecolor='white', format='tiff')
    plt.close()

def main():
    # 读取数据
    df = pd.read_csv('/Users/caijianyu/go/src/code.byted.org/argos/gbd/data/raw/breast_cancer.csv')
    pop = pd.read_csv('/Users/caijianyu/go/src/code.byted.org/argos/gbd/data/raw/pop.csv')
    
    # 分离死亡和发病数据
    deaths_data = df[df['measure_name'] == 'Deaths']
    incidence_data = df[df['measure_name'] == 'Incidence']
    
    # 计算中国的MIR
    china_1990 = calculate_mir(deaths_data, incidence_data, pop, 'China', 1990)
    china_2021 = calculate_mir(deaths_data, incidence_data, pop, 'China', 2021)  # 修改年份
    
    # 计算G20的MIR
    g20_1990 = calculate_mir(deaths_data, incidence_data, pop, 'G20', 1990)
    g20_2021 = calculate_mir(deaths_data, incidence_data, pop, 'G20', 2021)  # 修改年份
    
    # 合并数据并计算AAPC
    china_results = pd.merge(china_1990, china_2021, on=['Characteristics', 'Sex'], suffixes=('_1990', '_2021'))
    g20_results = pd.merge(g20_1990, g20_2021, on=['Characteristics', 'Sex'], suffixes=('_1990', '_2021'))
    
    # 计算AAPC
    def calculate_aapc(row):
        start_val = row['MIR_1990']
        end_val = row['MIR_2021']  # 修改年份
        start_lower = row['MIR_lower_1990']
        end_lower = row['MIR_lower_2021']
        start_upper = row['MIR_upper_1990']
        end_upper = row['MIR_upper_2021']
        years = 31  # 1990-2021，修改年数
        
        if start_val <= 0 or end_val <= 0:
            return "0.00 (0.00-0.00)"
        if start_val == end_val:
            return "0.00 (0.00-0.00)"
        
        # 计算主值AAPC
        result = ((end_val/start_val)**(1/years) - 1) * 100
        
        # 计算下限AAPC（使用上限值除以下限值得到最小变化率）
        lower_result = ((end_lower/start_upper)**(1/years) - 1) * 100 if start_upper > 0 else 0
        
        # 计算上限AAPC（使用下限值除以上限值得到最大变化率）
        upper_result = ((end_upper/start_lower)**(1/years) - 1) * 100 if start_lower > 0 else 0
        
        return f"{result:.2f} ({lower_result:.2f}-{upper_result:.2f})"
    
    # 添加这两行代码来应用calculate_aapc函数
    china_results['AAPC'] = china_results.apply(calculate_aapc, axis=1)
    g20_results['AAPC'] = g20_results.apply(calculate_aapc, axis=1)
    
    # 绘制表格
    plot_mir_table(china_results, g20_results)
    
    # 创建用于Excel输出的数据框
    excel_data = []
    
    # 添加China数据
    for _, row in china_results.iterrows():
        excel_data.append({
            'Region': 'China',
            'Characteristics': row['Characteristics'],
            'Sex': row['Sex'],
            'MIR_1990': row['MIR_1990'],
            'MIR_1990_lower': row['MIR_lower_1990'],
            'MIR_1990_upper': row['MIR_upper_1990'],
            'MIR_2021': row['MIR_2021'],
            'MIR_2021_lower': row['MIR_lower_2021'],
            'MIR_2021_upper': row['MIR_upper_2021'],
            'AAPC': row['AAPC']
        })
    
    # 添加G20数据
    for _, row in g20_results.iterrows():
        excel_data.append({
            'Region': 'G20',
            'Characteristics': row['Characteristics'],
            'Sex': row['Sex'],
            'MIR_1990': row['MIR_1990'],
            'MIR_1990_lower': row['MIR_lower_1990'],
            'MIR_1990_upper': row['MIR_upper_1990'],
            'MIR_2021': row['MIR_2021'],
            'MIR_2021_lower': row['MIR_lower_2021'],
            'MIR_2021_upper': row['MIR_upper_2021'],
            'AAPC': row['AAPC']
        })
    
    # 创建Excel表格
    excel_df = pd.DataFrame(excel_data)
    excel_path = '/Users/caijianyu/go/src/code.byted.org/argos/gbd/output/mir_table.xlsx'
    excel_df.to_excel(excel_path, index=False)
    
    print("MIR分析结果已保存为图表：/Users/caijianyu/go/src/code.byted.org/argos/gbd/output/mir_table.png")
    print("MIR分析结果已保存为Excel表格：/Users/caijianyu/go/src/code.byted.org/argos/gbd/output/mir_table.xlsx")

if __name__ == "__main__":
    main()