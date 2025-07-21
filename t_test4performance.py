import pandas as pd
import numpy as np
from scipy import stats

# 读取CSV文件
df = pd.read_csv('LLM4CDSR/log/amazon/One4All.csv')
# 提取基准模型(raw)的数据
baseline = df[df['model_name'] == 'aug1'].iloc[0]

# 提取增强模型(aug1,2,3)的数据
#aug_models = df[df['model_name'].str.startswith('aug')]
aug_models = df[df['model_name'].str.startswith('bcl_MLP_A')]
# 定义要测试的指标
metrics = ['NDCG@10', 'HR@10', 'NDCG@10_A', 'HR@10_A', 'NDCG@10_B', 'HR@10_B']

def perform_paired_ttest(aug_values, baseline_value, metric_name):
    """
    执行单边配对t检验
    """
    # 创建与aug_values等长的baseline数组
    baseline_array = np.repeat(baseline_value, len(aug_values))
    
    # 计算t统计量和p值（配对t检验）
    t_stat, p_val = stats.ttest_rel(aug_values, baseline_array)
    
    # 转换为单边检验的p值
    p_val_one_sided = p_val / 2 if t_stat > 0 else 1 - p_val / 2
    
    # 计算平均提升百分比
    improvement = ((aug_values.mean() - baseline_value) / baseline_value) * 100
    
    return {
        'metric': metric_name,
        't_statistic': t_stat,
        'p_value': p_val_one_sided,
        'significant': p_val_one_sided <= 0.05,
        'improvement': improvement,
        'baseline': baseline_value,
        'enhanced_mean': aug_values.mean(),
        'enhanced_std': aug_values.std(),
        'individual_improvements': [
            ((val - baseline_value) / baseline_value) * 100 
            for val in aug_values
        ]
    }

# 存储所有指标的检验结果
results = []

# 对每个指标进行检验
for metric in metrics:
    # 获取增强模型的值
    aug_values = aug_models[metric].values
    # 获取基准模型的值
    baseline_value = baseline[metric]
    
    # 执行检验
    result = perform_paired_ttest(aug_values, baseline_value, metric)
    results.append(result)

# 打印结果
print("配对t检验统计结果：")
print("-" * 80)
for result in results:
    print(f"\n指标: {result['metric']}")
    print(f"基准值: {result['baseline']:.4f}")
    print(f"增强模型均值±标准差: {result['enhanced_mean']:.4f}±{result['enhanced_std']:.4f}")
    print(f"平均提升: {result['improvement']:.2f}%")
    print("各次实验提升：")
    for i, imp in enumerate(result['individual_improvements'], 1):
        print(f"  实验{i}: {imp:.2f}%")
    print(f"t统计量: {result['t_statistic']:.4f}")
    print(f"p值: {result['p_value']:.4f}")
    print(f"是否显著: {'是' if result['significant'] else '否'} (α=0.05)")
    print("-" * 40)

# 将结果保存为更详细的DataFrame
detailed_results = []
for result in results:
    row = {
        'metric': result['metric'],
        'baseline': result['baseline'],
        'enhanced_mean': result['enhanced_mean'],
        'enhanced_std': result['enhanced_std'],
        'improvement_percent': result['improvement'],
        't_statistic': result['t_statistic'],
        'p_value': result['p_value'],
        'significant': result['significant']
    }
    # 添加各次实验的具体提升
    for i, imp in enumerate(result['individual_improvements'], 1):
        row[f'improvement_exp_{i}'] = imp
    detailed_results.append(row)

# 保存详细结果
results_df = pd.DataFrame(detailed_results)
results_df.to_csv('paired_ttest_results.csv', index=False)