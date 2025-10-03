import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from main import process_data  # 添加这行导入

def calculate_age_distribution(data, measure_type, sex_name=None):
    """计算年龄分布百分比"""
    # 选择特定年份 - 保持原来的年份设置
    years = [1990, 1995, 2000, 2005, 2010, 2015, 2021]
    age_groups = ['5-14 years', '15-49 years', '50-69 years', '70+ years']
    
    # 筛选数据
    filtered_data = data[
        (data['year'].isin(years)) &
        (data['age_name'].isin(age_groups))
    ].copy()
    
    # 如果指定了性别，进一步筛选
    if sex_name:
        filtered_data = filtered_data[filtered_data['sex_name'] == sex_name]
    
    # 计算每年各年龄组的百分比
    results = []
    for year in years:
        year_data = filtered_data[filtered_data['year'] == year]
        total = year_data['val'].sum()
        
        for age_group in age_groups:
            age_val = year_data[year_data['age_name'] == age_group]['val'].sum()
            percentage = (age_val / total * 100) if total > 0 else 0
            
            results.append({
                'year': year,
                'age_group': age_group.replace(' years', ''),
                'percentage': percentage
            })
    
    return pd.DataFrame(results)

def plot_age_distribution(df_incidence, df_population):
    # 创建6x2的子图布局：左栏Incidence，右栏Mortality
    fig, axes = plt.subplots(6, 2, figsize=(16, 18))
    
    # 调整子图之间的间距
    plt.subplots_adjust(
        left=0.08,
        right=0.85,  # 为右上角图例留出空间
        top=0.92,
        bottom=0.05,
        wspace=0.25,
        hspace=0.3
    )
    
    # 设置颜色映射
    colors = {
        '5-14': '#FF6B6B',
        '15-49': '#4ECDC4', 
        '50-69': '#45B7D1',
        '70+': '#96CEB4'
    }
    
    # 年份定义 - 保持原来的年份
    years = [1990, 1995, 2000, 2005, 2010, 2015, 2021]
    sexes = ['Both', 'Male', 'Female']
    locations = ['China', 'G20']
    measures = ['Incidence', 'Deaths']
    
    # 子图标签
    subplot_labels = [['A', 'B'], ['C', 'D']]
    
    # 为每个地区、指标和性别组合创建子图
    for loc_idx, location in enumerate(locations):
        for measure_idx, measure in enumerate(measures):
            col = measure_idx  # 左栏Incidence(0)，右栏Mortality(1)
            
            # 处理数据
            measure_data = df_incidence[df_incidence['measure_name'] == measure]
            data = process_data(measure_data, df_population, location)
            
            # 为每个性别创建子图
            for sex_idx, sex in enumerate(sexes):
                row = loc_idx * 3 + sex_idx  # China: 0-2行, G20: 3-5行
                ax = axes[row, col]
                
                dist_data = calculate_age_distribution(data, measure, sex)
                
                # 准备数据
                bottom = np.zeros(len(years))
                y_pos = np.arange(len(years))
                
                for age_group in ['5-14', '15-49', '50-69', '70+']:
                    age_data = dist_data[dist_data['age_group'] == age_group]
                    if not age_data.empty:
                        values = age_data['percentage'].values
                        if len(values) == len(years):
                            bars = ax.barh(y_pos, values, left=bottom,
                                         color=colors[age_group],
                                         height=0.6)
                            
                            # 添加数值标签
                            for i, (b, val) in enumerate(zip(bars, values)):
                                if val > 3:  # 只在足够大的区域显示标签
                                    x_pos = bottom[i] + val/2
                                    ax.text(x_pos, y_pos[i],
                                          f'{val:.1f}',
                                          ha='center',
                                          va='center',
                                          color='black',
                                          fontsize=8)
                            
                            bottom += values
                
                # 设置坐标轴
                ax.set_yticks(y_pos)
                ax.set_yticklabels(years)
                ax.set_ylim(-0.5, len(years) - 0.5)
                ax.set_xlim(0, 100)
                
                # 添加网格线
                ax.grid(True, linestyle='--', alpha=0.3, axis='x')
                
                # 设置边框
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                
                # 设置标题（只在第一行显示）
                if row == loc_idx * 3:  # 每个地区的第一行
                    label = subplot_labels[loc_idx][measure_idx]
                    title = 'Incidence' if measure == 'Incidence' else 'Mortality'
                    ax.set_title(f'{label}\n{title}', loc='left', pad=10, fontsize=12, weight='bold')
                
                # 添加性别标签（在左侧）
                if col == 0:  # 只在左栏添加性别标签
                    ax.text(-15, len(years)/2, sex, rotation=90, 
                           ha='center', va='center', fontsize=10, weight='bold')
                
                # 添加地区标签（在最左侧中间位置）
                if col == 0 and sex_idx == 1:  # 在每个地区的中间性别位置
                    ax.text(-30, len(years)/2, location, rotation=90, 
                           ha='center', va='center', fontsize=12, weight='bold')
    
    # 创建图例并放在右上角（移到循环外部）
    handles = [
        plt.Rectangle((0,0),1,1, color='#FF6B6B', label='5-14 years'),
        plt.Rectangle((0,0),1,1, color='#4ECDC4', label='15-49 years'),
        plt.Rectangle((0,0),1,1, color='#45B7D1', label='50-69 years'),
        plt.Rectangle((0,0),1,1, color='#96CEB4', label='70+ years')
    ]
    
    # 将图例放在右上角
    fig.legend(handles=handles,
              loc='upper right',
              bbox_to_anchor=(0.98, 0.98),
              title='Age groups',
              frameon=True,
              fancybox=True,
              shadow=True)
    
    plt.tight_layout()
    
    # 保存图片（移到循环外部）
    plt.savefig('/Users/caijianyu/go/src/code.byted.org/argos/gbd/output/age_distribution_by_sex.tif',
                format='tiff',
                dpi=600,
                bbox_inches='tight',
                pad_inches=0.2,
                facecolor='white',
                edgecolor='none')
    
    plt.show()  # 显示图形
    plt.close()  # 最后关闭图形

def main():
    # 读取数据
    df = pd.read_csv('/Users/caijianyu/go/src/code.byted.org/argos/gbd/data/raw/breast_cancer.csv')
    pop = pd.read_csv('/Users/caijianyu/go/src/code.byted.org/argos/gbd/data/raw/pop.csv')
    
    # 绘制年龄分布图
    plot_age_distribution(df, pop)
    print("按性别分组的年龄分布图已保存：age_distribution_by_sex.tif")

if __name__ == "__main__":
    main()