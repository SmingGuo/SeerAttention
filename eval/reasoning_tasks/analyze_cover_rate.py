import re

def analyze_coverage_rates(file_path):
    """
    从文件中读取内容，分析覆盖率数据，找出比全局平均值高/低10%以上、15%以上和20%以上的(层, 头)
    
    Args:
        file_path (str): 包含覆盖率数据的文本文件路径
    """
    # 1. 读取文件内容
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            file_content = file.read()
    except FileNotFoundError:
        print(f"错误：文件 '{file_path}' 未找到。")
        return
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return
        
    # 2. 提取全局加权平均覆盖率
    lines = file_content.strip().split('\n')
    global_avg_rate_line = None
    for line in reversed(lines):  # 从最后一行往前找，因为全局平均值通常在文件末尾
        if '全局加权平均覆盖率' in line:
            global_avg_rate_line = line
            break
    
    if not global_avg_rate_line:
        print("错误：找不到全局加权平均覆盖率。")
        return

    try:
        global_avg_rate = float(global_avg_rate_line.split(':')[1].strip())
    except (IndexError, ValueError):
        print("错误：无法解析全局加权平均覆盖率。")
        return

    # 3. 计算所有阈值
    thresholds = {
        'below_10': global_avg_rate * 0.90,  # 低于10%以上
        'below_15': global_avg_rate * 0.85,  # 低于15%以上
        'below_20': global_avg_rate * 0.80,  # 低于20%以上
        'above_10': global_avg_rate * 1.10,  # 高于10%以上
        'above_15': global_avg_rate * 1.15,  # 高于15%以上
        'above_20': global_avg_rate * 1.20,  # 高于20%以上
    }

    # 4. 初始化结果列表
    results = {
        'below_10': [],  # 低于10%以上
        'below_15': [],  # 低于15%以上
        'below_20': [],  # 低于20%以上
        'above_10': [],  # 高于10%以上
        'above_15': [],  # 高于15%以上
        'above_20': [],  # 高于20%以上
    }

    # 5. 解析每个(层, 头)的覆盖率
    # 使用正则表达式匹配 "层 X, 头 Y: Z.ZZZZ" 格式
    pattern = re.compile(r"层\s*(\d+),\s*头\s*(\d+):\s*([\d.]+)")

    for line in lines:
        match = pattern.search(line)
        if match:
            try:
                layer = int(match.group(1))
                head = int(match.group(2))
                rate = float(match.group(3))
                layer_head_pair = (layer, head)

                # 6. 与阈值比较并分类
                # 所有低于10%以上的点（包括15%、20%）
                if rate < thresholds['below_10']:
                    results['below_10'].append(layer_head_pair)
                    
                # 所有低于15%以上的点（包括20%）
                if rate < thresholds['below_15']:
                    results['below_15'].append(layer_head_pair)
                    
                # 所有低于20%以上的点
                if rate < thresholds['below_20']:
                    results['below_20'].append(layer_head_pair)
                    
                # 所有高于10%以上的点（包括15%、20%）
                if rate > thresholds['above_10']:
                    results['above_10'].append(layer_head_pair)
                    
                # 所有高于15%以上的点（包括20%）
                if rate > thresholds['above_15']:
                    results['above_15'].append(layer_head_pair)
                    
                # 所有高于20%以上的点
                if rate > thresholds['above_20']:
                    results['above_20'].append(layer_head_pair)

            except (ValueError, IndexError):
                # 跳过格式不正确的行
                continue

    # 7. 分组输出结果（每10项一行）
    def format_output(items, per_line=10):
        if not items:
            return ["无"]
            
        sorted_items = sorted(items)
        output_lines = []
        for i in range(0, len(sorted_items), per_line):
            group = sorted_items[i:i+per_line]
            line = ", ".join([f"({l},{h})" for l, h in group])
            output_lines.append(line)
        return output_lines

    # 8. 打印结果
    print(f"全局加权平均覆盖率: {global_avg_rate:.4f}\n")
    
    # 低于结果
    print(f"所有覆盖率低于全局平均值10%以上 (阈值: < {thresholds['below_10']:.4f}) 的 (层, 头) (共 {len(results['below_10'])} 个):")
    for line in format_output(results['below_10']):
        print(line)
    print()

    # 高于结果
    print(f"所有覆盖率高于全局平均值10%以上 (阈值: > {thresholds['above_10']:.4f}) 的 (层, 头) (共 {len(results['above_10'])} 个):")
    for line in format_output(results['above_10']):
        print(line)
    print()
    
    print(f"所有覆盖率低于全局平均值15%以上 (阈值: < {thresholds['below_15']:.4f}) 的 (层, 头) (共 {len(results['below_15'])} 个):")
    for line in format_output(results['below_15']):
        print(line)
    print()
    
    print(f"所有覆盖率高于全局平均值15%以上 (阈值: > {thresholds['above_15']:.4f}) 的 (层, 头) (共 {len(results['above_15'])} 个):")
    for line in format_output(results['above_15']):
        print(line)
    print()
    
    print(f"所有覆盖率低于全局平均值20%以上 (阈值: < {thresholds['below_20']:.4f}) 的 (层, 头) (共 {len(results['below_20'])} 个):")
    for line in format_output(results['below_20']):
        print(line)
    print()
    print(f"所有覆盖率高于全局平均值20%以上 (阈值: > {thresholds['above_20']:.4f}) 的 (层, 头) (共 {len(results['above_20'])} 个):")
    for line in format_output(results['above_20']):
        print(line)
    print("-" * 80)
    
    below_total = len(results['below_10']) + len(results['below_15']) + len(results['below_20'])
    above_total = len(results['above_10']) + len(results['above_15']) + len(results['above_20'])
    print(f"\n总波动项: {below_total + above_total} 个 (所有大于10%偏差的点)")
    print(f"总异常项: {len(results['below_15']) + len(results['above_15'])} 个 (所有大于15%偏差的点)")
    print(f"总严重异常项: {len(results['below_20']) + len(results['above_20'])} 个 (所有大于20%偏差的点)")


# 使用示例
if __name__ == "__main__":
    file_path = "/home/v-shumingguo/gsm_blob/results/profile/Qwen3-14B/livecodebench_6k_cover_rate.txt"  # 替换为你的文件路径
    analyze_coverage_rates(file_path)