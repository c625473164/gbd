import pandas as pd
import numpy as np

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

def calculate_country_metrics(df_incidence, df_population, country):
    """计算单个国家的指标"""
    # 处理发病数据
    country_incidence = df_incidence[
        (df_incidence['location_name'] == country) &
        (df_incidence['cause_name'] == 'Breast cancer')
    ].copy()
    
    # 处理人口数据
    country_population = df_population[
        (df_population['location'] == country)
    ].copy()
    
    # 计算1990年和2021年的ASIR和ASMR
    metrics = {}
    for year in [1990, 2021]:
        year_data = country_incidence[country_incidence['year'] == year]
        year_pop = country_population[country_population['year'] == year]
        
        # 计算ASIR
        if 'Incidence' in year_data['measure_name'].values:
            asir_data = year_data[year_data['measure_name'] == 'Incidence']
            metrics[f'asir_{year}'] = calculate_standardized_rate(asir_data, year_pop)
        
        # 计算ASMR
        if 'Deaths' in year_data['measure_name'].values:
            asmr_data = year_data[year_data['measure_name'] == 'Deaths']
            metrics[f'asmr_{year}'] = calculate_standardized_rate(asmr_data, year_pop)
    
    # 计算AAPC
    metrics['asir_aapc'] = calculate_aapc(metrics.get('asir_1990', 0), metrics.get('asir_2021', 0))
    metrics['asmr_aapc'] = calculate_aapc(metrics.get('asmr_1990', 0), metrics.get('asmr_2021', 0))
    
    return metrics

def calculate_standardized_rate(data, population_data):
    """计算年龄标准化率"""
    try:
        # 世界标准人口权重
        age_weights = {
            '5-14 years': 0.17290,
            '15-49 years': 0.52010,
            '50-69 years': 0.16600,
            '70+ years': 0.05275
        }
        
        standardized_rate = 0
        total_weight = sum(age_weights.values())
        
        for age_group, weight in age_weights.items():
            age_data = data[data['age_name'] == age_group]
            age_pop = population_data[population_data['age'] == age_group]
            
            if not age_data.empty and not age_pop.empty:
                cases = age_data['val'].sum()
                population = age_pop['val'].sum()
                if population > 0:
                    crude_rate = (cases / population) * 100000
                    standardized_rate += crude_rate * weight
        
        return standardized_rate / total_weight
    except Exception as e:
        print(f"计算标准化率时出错: {e}")
        return 0.00

def main():
    # 读取数据
    df_incidence = pd.read_csv('/Users/caijianyu/go/src/code.byted.org/argos/gbd/data/raw/breast_cancel_1.csv')
    df_population = pd.read_csv('/Users/caijianyu/go/src/code.byted.org/argos/gbd/data/raw/pop_g20.csv')
    
    # 获取所有国家列表
    countries = df_incidence['location_name'].unique()
    
    # 存储所有国家的结果
    results = []
    for country in countries:
        metrics = calculate_country_metrics(df_incidence, df_population, country)
        metrics['country'] = country
        results.append(metrics)
    
    # 转换为DataFrame并保存
    results_df = pd.DataFrame(results)
    results_df.to_csv('/Users/caijianyu/go/src/code.byted.org/argos/gbd/output/country_metrics.csv', index=False)
    print("国家级指标计算完成，结果已保存到 country_metrics.csv")

if __name__ == "__main__":
    main()