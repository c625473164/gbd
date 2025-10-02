import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from main import process_data

def calculate_standardized_rate(data, population, type='ASIR'):
    """计算年龄标准化率（ASIR或ASMR）
    
    Args:
        data: 包含 val, population, age_name 列的 DataFrame
        population: 人口数据
        type: 计算类型，'ASIR'（发病率）或'ASMR'（死亡率）
        
    Returns:
        float: 标准化率。当数据为空或出错时返回0
    """
    
    # WHO世界标准人口权重
    age_weights = {
        '5-14 years': 0.17290,
        '15-49 years': 0.52010,
        '50-69 years': 0.16600,
        '70+ years': 0.05275
    }
    
    # 确保数据不为空
    if data.empty:
        # 对于任何类型的空数据都返回0
        return 0
    
    # 检查必要的列是否存在
    required_columns = ['val', 'population', 'age_name']
    if not all(col in data.columns for col in required_columns):
        raise ValueError(f"缺少必要的列. 现有列: {data.columns.tolist()}")
    
    # 计算粗率
    crude_rates = (data['val'] / data['population']) * 100000

    # 计算标准化率
    standardized_rate = 0
    total_weight = sum(age_weights.values())
    for age_group, weight in age_weights.items():
        age_data = data[data['age_name'] == age_group]
        if not age_data.empty:
            crude_rate = crude_rates[age_data.index].mean()
            standardized_rate += crude_rate * weight
    
    result = standardized_rate / total_weight
    return result
    


def calculate_asr_mir(asir_data, asmr_data):
    """计算年龄标准化死亡发病比"""
    if asir_data == 0:
        return 0
    return (asmr_data / asir_data) 

def plot_trends(df_incidence, df_mortality, df_population):
    # 创建2×3×3的子图布局
    fig, axes = plt.subplots(6, 3, figsize=(20, 24))
    
    locations = ['China', 'G20']
    years = range(1990, 2021)
    titles = ['ASIR', 'ASMR', 'ASR of MIR']
    
    # 定义年龄组和颜色
    age_groups = ['All ages', '5-14 years', '15-49 years', '50-69 years', '70+ years']
    colors = {
        'All ages': 'black',
        '5-14 years': 'red',
        '15-49 years': 'green',
        '50-69 years': 'cyan',
        '70+ years': 'blue'
    }
    
    # 处理每个地区的数据
    for loc_idx, location in enumerate(locations):
        # 修改数据处理部分
        incidence_data = process_data(
            df_incidence[
                (df_incidence['measure_name'] == 'Incidence') & 
                (df_incidence['location_name'] == location)
            ].copy(), 
            df_population[
                (df_population['location'] == location) & 
                (df_population['year'].isin(years))
            ].copy(),
            location
        )
        
        mortality_data = process_data(
            df_mortality[
                (df_mortality['measure_name'] == 'Deaths') & 
                (df_mortality['location_name'] == location)
            ].copy(), 
            df_population[
                (df_population['location'] == location) & 
                (df_population['year'].isin(years))
            ].copy(),
            location
        )
        
        # 计算各项指标并按年龄组存储
        results_by_age = {}
        for age_group in age_groups:
            results_by_age[age_group] = []
            for year in years:
                year_inc = incidence_data[
                    (incidence_data['year'] == year) & 
                    (incidence_data['age_name'] == age_group)
                ]
                year_mort = mortality_data[
                    (mortality_data['year'] == year) & 
                    (mortality_data['age_name'] == age_group)
                ]
                
                for sex in ['Both', 'Male', 'Female']:
                    sex_inc = year_inc[year_inc['sex_name'] == sex]
                    sex_mort = year_mort[year_mort['sex_name'] == sex]
                    
                    try:
                        asir = calculate_standardized_rate(sex_inc, df_population, 'ASIR')
                        asmr = calculate_standardized_rate(sex_mort, df_population, 'ASMR')
                    except Exception as e:
                        print(f"\n计算错误详情:")
                        print(f"地区: {location}, 年份: {year}, 性别: {sex}, 年龄组: {age_group}")
                        print(f"发病数据形状: {sex_inc.shape}")
                        print(f"死亡数据形状: {sex_mort.shape}")
                        print(f"错误信息: {str(e)}")
                        asir, asmr = 0, 0
                    
                    asr_mir = calculate_asr_mir(asir, asmr)
                    
                    results_by_age[age_group].append({
                        'year': year,
                        'sex': sex,
                        'asir': asir,
                        'asmr': asmr,
                        'asr_mir': asr_mir
                    })
        
        # 绘制趋势图
        start_idx = loc_idx * 3
        for sex_idx, sex in enumerate(['Both', 'Male', 'Female']):
            row_idx = start_idx + sex_idx
            for col_idx, (title, measure) in enumerate(zip(titles, ['asir', 'asmr', 'asr_mir'])):
                ax = axes[row_idx, col_idx]
                
                # 按年龄组分别绘制
                for age_group in age_groups:
                    age_results = [r for r in results_by_age[age_group] if r['sex'] == sex]
                    if age_results:
                        years_data = [r['year'] for r in age_results]
                        values = [r[measure] for r in age_results]
                        ax.plot(years_data, values, 
                               color=colors[age_group], 
                               label=age_group,
                               linewidth=1.5)
                
                # 设置轴和网格
                ax.grid(True, alpha=0.3, linestyle='--')
                ax.set_xlabel('Year')
                ax.set_ylabel(title)
                
                # 设置标题
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
    
    # 调整布局
    plt.tight_layout()
    
    # 添加图例
    fig.legend(['All ages', '5-14 ages', '15-49 ages', '50-69 ages', '≥70 ages'],
              loc='center', bbox_to_anchor=(0.5, -0.02), ncol=5,
              facecolor='white', edgecolor='none')
    
    # 保存图片
    plt.savefig('/Users/caijianyu/go/src/code.byted.org/argos/gbd/output/trends.png',
                dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # 读取数据
    df = pd.read_csv('/Users/caijianyu/go/src/code.byted.org/argos/gbd/data/raw/breast_cancer.csv')
    pop = pd.read_csv('/Users/caijianyu/go/src/code.byted.org/argos/gbd/data/raw/pop.csv')
    
    # 绘制趋势图
    plot_trends(df, df, pop)  # 这里传入相同的df是因为它包含了发病和死亡的数据
    print("趋势图已保存为：/Users/caijianyu/go/src/code.byted.org/argos/gbd/output/trends.png")

if __name__ == "__main__":
    main()