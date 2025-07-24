import matplotlib.pyplot as plt
import json
import numpy as np

# 解析JSON数据
filename = '/home/v-shumingguo/gsm_blob/results/training_log/mixedkl_a1.0/seer_sparse_attn_8k.json'
with open(filename, 'r') as f:
    data = json.load(f)

attn_data = np.array(data['attn_data'])
n = len(attn_data)

# 验证概率和
print(f"概率和: {attn_data.sum():.6f}")  # 输出: 1.000000
print(f"数据点总数: {n}")

# 排序数据（降序）
# sorted_data = np.sort(attn_data)
sorted_data = np.sort(attn_data)[::-1]

# 使用cumsum计算累积概率
cumulative_probs = np.cumsum(sorted_data)

# 准备横轴（步数/数据点索引）
step_indices = np.arange(len(sorted_data))  # 0到n-1的索引
full_step_indices = np.arange(0, len(sorted_data))  # 准备完整的步数索引

# 绘制阶梯图（横轴改为步数）
plt.figure(figsize=(10, 6))
plt.step(full_step_indices, cumulative_probs, where='post', linewidth=2, color='blue')
plt.title('CDF Seer Sparse Reverse KL 8k Down', fontsize=14)
plt.xlabel('Step Index (Sorted Data Points)', fontsize=12)
plt.ylabel('Cumulative Probability', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)

# 标注关键特征
# 1. 标记零值结束位置
zero_count = sum(1 for x in sorted_data if x == 0)
if zero_count > 0:
    plt.axvline(x=zero_count-1, color='red', linestyle='--', alpha=0.6)
    plt.text(zero_count+1, 0.05, f'Zero values end\nat step {zero_count-1}', fontsize=10)


# 3. 标记90%累计概率位置（新增值标注）
ninety_idx = np.argmax(cumulative_probs >= 0.9)
ninety_value = sorted_data[ninety_idx]  # 获取对应的注意力权重值
plt.axvline(x=ninety_idx, color='purple', linestyle=':', alpha=0.7)
plt.text(ninety_idx+1, 0.8, 
         f'90% at {ninety_idx}\nPercentage:{ninety_idx/n*100:.2f}%\nValue: {ninety_value:.6f}', 
         fontsize=10)

# 4. 最后的数据点（顶部值）
plt.scatter([n-1], [1], color='red', zorder=3)
plt.annotate(f'Final step: {n-1}\nValue: {sorted_data[-1]:.4f}', 
             xy=(n-1, 1),
             xytext=(n-15, 0.8),
             arrowprops=dict(arrowstyle='->'))

save_path = filename.replace('.json', '_cdf_down.png')
plt.savefig(save_path, dpi=120)
plt.show()