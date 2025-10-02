import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def predict_2050_asr(df_2021, aapc):
    """基于2021年数据和AAPC预测2050年的ASR"""
    years = 29  # 2021到2050年的年数
    return df_2021 * (1 + aapc/100) ** years

def plot_prediction_table(china_df, g20_df):
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 调整图形尺寸以适应并排布局
    fig, ax = plt.subplots(figsize=(16, 10))  # 增加宽度，减少高度
    plt.subplots_adjust(top=0.9)  # 调整顶部边距
    ax.axis('tight')
    ax.axis('off')
    
    # 修改表头为并排布局
    headers = ['Characteristics', 
              'China\n2050 ASIR\nper 100000',
              'China\n2050 ASMR\nper 100000',
              'G20\n2050 ASIR\nper 100000',
              'G20\n2050 ASMR\nper 100000']
    
    table_data = []
    
    # 按性别和年龄组组织数据，不添加性别标题行
    for sex in ['Both', 'Male', 'Female']:
        # 获取该性别的数据
        china_sex_data = china_df[china_df['Sex'] == sex]
        g20_sex_data = g20_df[g20_df['Sex'] == sex]
        
        # 按特征排序（总体在前，然后是年龄组）
        characteristics_order = [sex, '5-14', '15-49', '50-69', '70+']
        
        for char in characteristics_order:
            china_row = china_sex_data[china_sex_data['Characteristics'] == char]
            g20_row = g20_sex_data[g20_sex_data['Characteristics'] == char]
            
            if not china_row.empty and not g20_row.empty:
                table_data.append([
                    char,
                    f"{china_row.iloc[0]['ASIR_2050']:.2f}",
                    f"{china_row.iloc[0]['ASMR_2050']:.2f}",
                    f"{g20_row.iloc[0]['ASIR_2050']:.2f}",
                    f"{g20_row.iloc[0]['ASMR_2050']:.2f}"
                ])
    
    table = ax.table(cellText=table_data,
                    colLabels=headers,
                    loc='center',
                    cellLoc='center',
                    colWidths=[0.25, 0.1875, 0.1875, 0.1875, 0.1875])
    
    # 设置表格样式
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    
    # 调整表头行的高度和样式
    for col in range(len(headers)):
        cell = table[0, col]
        cell.set_height(0.15)
        cell.set_facecolor('#E6E6E6')
        cell.set_text_props(weight='bold')
    
    # 设置Both、Male、Female行的样式（现在它们是数据行而不是标题行）
    sex_data_rows = []
    for i, row in enumerate(table_data):
        if row[0] in ['Both', 'Male', 'Female']:
            sex_data_rows.append(i + 1)  # +1 因为表头占第0行
    
    for idx in sex_data_rows:
        for col in range(len(headers)):
            cell = table[idx, col]
            cell.set_facecolor('#F0F0F0')
            cell.set_text_props(weight='bold')
    
    plt.title('The prediction of mortality and incidence ASR of breast cancer in 2050',
              fontsize=12)
    
    plt.savefig('/Users/caijianyu/go/src/code.byted.org/argos/gbd/output/prediction_table.tif',
                dpi=600, bbox_inches='tight', facecolor='white', format='tiff')
    plt.close()

def extract_main_value(value_str):
    """从包含置信区间的字符串中提取主要数值
    例如: '23.66 (18.23, 30.09)' -> 23.66
    """
    if isinstance(value_str, (int, float)):
        return float(value_str)
    
    # 转换为字符串并去除空格
    value_str = str(value_str).strip()
    
    # 如果包含括号，提取括号前的数值
    if '(' in value_str:
        main_value = value_str.split('(')[0].strip()
    else:
        main_value = value_str
    
    try:
        return float(main_value)
    except ValueError:
        print(f"无法转换数值: {value_str}")
        return 0.0

def process_prediction_data(incidence_df, mortality_df):
    """处理预测数据"""
    # 添加调试信息
    print("\n调试信息:")
    print("发病率数据框列名:", incidence_df.columns.tolist())
    print("死亡率数据框列名:", mortality_df.columns.tolist())
    
    results = []
    for location in ['China', 'G20']:
        print(f"\n处理 {location} 数据:")
        # 使用完整数据集，不按地区过滤
        inc_location = incidence_df.copy()
        mort_location = mortality_df.copy()
        
        for sex in ['Both', 'Male', 'Female']:
            print(f"\n处理 {sex} 性别数据:")
            # 按性别过滤
            inc_sex = inc_location[inc_location['Sex'] == sex]
            mort_sex = mort_location[mort_location['Sex'] == sex]
            print(f"发病率数据行数: {len(inc_sex)}")
            print(f"死亡率数据行数: {len(mort_sex)}")
            
            # 计算总体预测值
            if len(inc_sex) > 0 and len(mort_sex) > 0:
                try:
                    asir_2021 = extract_main_value(inc_sex[inc_sex['Characteristics'] == sex]['2021_asir'].iloc[0])
                    asmr_2021 = extract_main_value(mort_sex[mort_sex['Characteristics'] == sex]['2021_asir'].iloc[0])
                    inc_aapc = extract_main_value(inc_sex[inc_sex['Characteristics'] == sex]['AAPC'].iloc[0])
                    mort_aapc = extract_main_value(mort_sex[mort_sex['Characteristics'] == sex]['AAPC'].iloc[0])
                    
                    asir_2050 = predict_2050_asr(asir_2021, inc_aapc)
                    asmr_2050 = predict_2050_asr(asmr_2021, mort_aapc)
                    
                    results.append({
                        'Location': location,
                        'Characteristics': sex,
                        'Sex': sex,
                        'ASIR_2050': asir_2050,
                        'ASMR_2050': asmr_2050
                    })
                except Exception as e:
                    print(f"处理{location}-{sex}总体数据时出错:", e)
            
            # 计算各年龄组预测值
            for age_group in ['5-14', '15-49', '50-69', '70+']:
                inc_age = inc_sex[inc_sex['Characteristics'] == age_group]
                mort_age = mort_sex[mort_sex['Characteristics'] == age_group]
                
                if len(inc_age) > 0 and len(mort_age) > 0:
                    try:
                        asir_2021 = extract_main_value(inc_age['2021_asir'].iloc[0])
                        asmr_2021 = extract_main_value(mort_age['2021_asir'].iloc[0])
                        inc_aapc = extract_main_value(inc_age['AAPC'].iloc[0])
                        mort_aapc = extract_main_value(mort_age['AAPC'].iloc[0])
                        
                        asir_2050 = predict_2050_asr(asir_2021, inc_aapc)
                        asmr_2050 = predict_2050_asr(asmr_2021, mort_aapc)
                        
                        results.append({
                            'Location': location,
                            'Characteristics': age_group,
                            'Sex': sex,
                            'ASIR_2050': asir_2050,
                            'ASMR_2050': asmr_2050
                        })
                    except Exception as e:
                        print(f"处理{location}-{sex}-{age_group}数据时出错:", e)
    
    return pd.DataFrame(results)

def main():
    # 读取数据
    df = pd.read_csv('/Users/caijianyu/go/src/code.byted.org/argos/gbd/data/raw/breast_cancer.csv')
    pop = pd.read_csv('/Users/caijianyu/go/src/code.byted.org/argos/gbd/data/raw/pop.csv')
    
    # 分别处理发病率和死亡率数据
    from main import process_data, create_result_table
    
    # 处理发病率数据
    incidence_data = df[df['measure_name'] == 'Incidence']
    china_inc = process_data(incidence_data, pop, 'China')
    g20_inc = process_data(incidence_data, pop, 'G20')
    china_inc_results = create_result_table(china_inc)
    g20_inc_results = create_result_table(g20_inc)
    
    # 处理死亡率数据
    mortality_data = df[df['measure_name'] == 'Deaths']
    china_mort = process_data(mortality_data, pop, 'China')
    g20_mort = process_data(mortality_data, pop, 'G20')
    china_mort_results = create_result_table(china_mort)
    g20_mort_results = create_result_table(g20_mort)
    print(123, china_inc_results)
    print(123, china_mort_results)
    # 合并数据并预测2050年结果
    china_results = process_prediction_data(china_inc_results, china_mort_results)
    g20_results = process_prediction_data(g20_inc_results, g20_mort_results)
    
    
    # 绘制预测表格
    plot_prediction_table(
        china_results[china_results['Location'] == 'China'],
        g20_results[g20_results['Location'] == 'G20']
    )
    
    print("2050年预测结果已保存为图表：/Users/caijianyu/go/src/code.byted.org/argos/gbd/output/prediction_table.tif")

if __name__ == "__main__":
    main()