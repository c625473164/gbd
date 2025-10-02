import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from main import process_data  # 添加这行导入

def calculate_age_distribution(data, measure_type):
    """计算年龄分布百分比"""
    # 选择特定年份
    years = [1990, 1995, 2000, 2005, 2010, 2015, 2021]
    age_groups = ['5-14 years', '15-49 years', '50-69 years', '70+ years']
    
    # 筛选数据
    filtered_data = data[
        (data['year'].isin(years)) &
        (data['age_name'].isin(age_groups))
    ].copy()
    
    # 计算每年各年龄组的百分比
    results = []
    for year in years:
        year_data = filtered_data[filtered_data['year'] == year]
        total = year_data['val'].sum()
        
        for age_group in age_groups:
            # 修改这里：使用 'age_name' 而不是 'age_group'
            age_val = year_data[year_data['age_name'] == age_group]['val'].sum()
            percentage = (age_val / total * 100) if total > 0 else 0
            
            results.append({
                'year': year,
                'age_group': age_group.replace(' years', ''),
                'percentage': percentage
            })
    
    return pd.DataFrame(results)

def plot_age_distribution(df_incidence, df_population):
    # 增加图形宽度
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))  # 从(12, 10)改为(15, 10)
    
    # 调整子图之间的间距
    plt.subplots_adjust(
        left=0.15,    # 左边距
        right=0.85,   # 减小右边距，给图例留更多空间
        top=0.95,     
        bottom=0.05,  
        wspace=0.25,  # 略微增加水平间距
        hspace=0.2    
    )
    
    # 设置颜色映射
    colors = {
        '5-14': '#FF9999',
        '15-49': '#90EE90',
        '50-69': '#87CEEB',
        '70+': '#B19CD9'
    }
    
    # 添加年份定义
    years = [1990, 1995, 2000, 2005, 2010, 2015, 2021]
    
    # 处理数据并绘图
    locations = ['China', 'G20']
    measures = ['Incidence', 'Deaths']
    axes = [(ax1, 'China', 'Incidence'), 
            (ax2, 'China', 'Deaths'),
            (ax3, 'G20', 'Incidence'), 
            (ax4, 'G20', 'Deaths')]
    
    for ax, location, measure in axes:
        print(f"\n处理 {location} - {measure} 数据")
        # 处理数据
        measure_data = df_incidence[df_incidence['measure_name'] == measure]
        data = process_data(measure_data, df_population, location)
        dist_data = calculate_age_distribution(data, measure)
        
        # 准备数据
        bottom = np.zeros(len(years))
        # 使用数组索引作为位置，而不是年份数值
        y_pos = np.arange(len(years))  # [0, 1, 2, 3, 4, 5, 6]
        
        for age_group in ['5-14', '15-49', '50-69', '70+']:
            age_data = dist_data[dist_data['age_group'] == age_group]
            if not age_data.empty:
                values = age_data['percentage'].values
                if len(values) == len(years):
                    bars = ax.barh(y_pos, values, left=bottom, 
                                 label=age_group, 
                                 color=colors[age_group], 
                                 height=0.5)  # 调整高度
                    
                    # 标签位置也使用索引位置
                    for i, (b, val) in enumerate(zip(bars, values)):
                        if val > 0:
                            x_pos = bottom[i] + val/2
                            ax.text(x_pos, y_pos[i], 
                                  f'{val:.1f}',
                                  ha='center', 
                                  va='center',
                                  color='black',
                                  fontsize=8)
                    
                    bottom += values
        
        # 设置y轴为类别型
        # 移除重复的y轴设置，只保留一组
        ax.set_yticks(y_pos)
        ax.set_yticklabels(years)
        ax.set_ylim(-0.5, len(years) - 0.5)
        ax.set_xlim(0, 100)
        
        # 添加网格线
        ax.grid(True, linestyle='--', alpha=0.1)
        
        # 设置边框显示
        ax.spines['top'].set_visible(False)  # 隐藏上边框
        ax.spines['right'].set_visible(False)  # 隐藏右边框
        ax.spines['left'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        
        # 删除这行，它会覆盖上面的边框设置
        # for spine in ax.spines.values():
        #     spine.set_visible(True)
    
    # 修改图例设置
    handles = [
        # plt.Rectangle((0,0),1,1, color='#FF9999', label='5-14 years'),
        plt.Rectangle((0,0),1,1, color='#90EE90', label='15-49 years'),
        plt.Rectangle((0,0),1,1, color='#87CEEB', label='50-69 years'),
        plt.Rectangle((0,0),1,1, color='#B19CD9', label='70+ years')
    ]
    
    # 调整图例位置
    fig.legend(handles=handles, 
              loc='upper right',
              bbox_to_anchor=(1.1, 0.98),  # 调整图例位置
              title='Age groups',
              frameon=True,
              edgecolor='black')
    

    # 移除每个子图的单独图例
    for ax in [ax1, ax2, ax3, ax4]:
        ax.get_legend().remove() if ax.get_legend() else None
    
    # 调整布局
    plt.tight_layout(pad=1.5)
    
    # 删除这段重复的代码
    # handles, labels = ax1.get_legend_handles_labels()
    # fig.legend(handles, labels, loc='center right', title='Age groups')
    
    # 调整布局
    # plt.tight_layout()
     # 设置子图标题
    ax1.set_title('A\nIncidence', loc='left', pad=10)
    ax2.set_title('B\nMortality', loc='left', pad=10)
    ax3.set_title('C', loc='left', pad=10)
    ax4.set_title('D', loc='left', pad=10)
    
    # 为每个子图添加标签
    for ax in [ax1, ax2, ax3, ax4]:
        # 只为左侧两个子图（A和C）添加性别标签
        if ax in [ax1, ax3]:
            # ax.text(-25, 5, 'Both', rotation=90, ha='center', va='bottom', fontsize=12)
            # ax.text(-25, 3, 'Male', rotation=90, ha='center', va='bottom', fontsize=12)
            # ax.text(-25, 1, 'Female', rotation=90, ha='center', va='bottom', fontsize=12)
            
            # 只为左侧两个子图添加地区标签
            if ax == ax1:
                ax.text(-10, 3, 'China', rotation=90, ha='center', va='center', fontsize=12)
            else:
                ax.text(-10, 3, 'G20', rotation=90, ha='center', va='center', fontsize=12)
        
        # 设置刻度标签字体大小
        ax.tick_params(axis='both', labelsize=11)
    
    # 设置标题字体大小
    ax1.set_title('A\nIncidence', loc='left', pad=10, fontsize=12)
    ax2.set_title('B\nMortality', loc='left', pad=10, fontsize=12)
    ax3.set_title('C', loc='left', pad=10, fontsize=12)
    ax4.set_title('D', loc='left', pad=10, fontsize=12)
    
    # 调整布局（在添加所有元素之后）
    plt.tight_layout()
    
    # 保存符合所有DPI要求的TIF格式图片（使用1200 dpi同时满足所有条件）
    plt.savefig('/Users/caijianyu/go/src/code.byted.org/argos/gbd/output/age_distribution.tif',
                format='tiff',
                dpi=1200, 
                bbox_inches='tight',
                pad_inches=0.2,
                facecolor='white',
                edgecolor='none')
    
    plt.close()

def main():
    # 读取数据
    df = pd.read_csv('/Users/caijianyu/go/src/code.byted.org/argos/gbd/data/raw/breast_cancer.csv')
    pop = pd.read_csv('/Users/caijianyu/go/src/code.byted.org/argos/gbd/data/raw/pop.csv')
    
    # 绘制年龄分布图
    plot_age_distribution(df, pop)
    print("年龄分布图已保存为TIF格式：age_distribution.tif (1200 dpi)")

if __name__ == "__main__":
    main()