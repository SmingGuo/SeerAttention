import re
import json
import os

def analyze_and_save_coverage(file_path, output_file):
    """
    分析覆盖率数据并将所有结果保存到单一JSON文件
    
    Args:
        file_path (str): 包含覆盖率数据的文本文件路径
        output_file (str): 输出JSON文件路径
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            file_content = file.read()
    except FileNotFoundError:
        print(f"错误：文件 '{file_path}' 未找到。")
        return
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return
        
    lines = file_content.strip().split('\n')
    global_avg_rate_line = None
    for line in reversed(lines):
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

    below_threshold = global_avg_rate * 0.90
    above_threshold = global_avg_rate * 1.10

    # 存储完整信息用于后续处理
    below_data = []  # (layer, head, rate, deviation)
    above_data = []  # (layer, head, rate, deviation)
    all_pairs = []   # (layer, head)

    pattern = re.compile(r"层\s*(\d+),\s*头\s*(\d+):\s*([\d.]+)")

    for line in lines:
        match = pattern.search(line)
        if match:
            try:
                layer = int(match.group(1))
                head = int(match.group(2))
                rate = float(match.group(3))
                all_pairs.append((layer, head))

                if rate < below_threshold:
                    # 计算偏离程度（阈值以下，偏离程度 = 阈值 - 覆盖率）
                    deviation = below_threshold - rate
                    below_data.append((layer, head, rate, deviation))
                    
                if rate > above_threshold:
                    # 计算偏离程度（阈值以上，偏离程度 = 覆盖率 - 阈值）
                    deviation = rate - above_threshold
                    above_data.append((layer, head, rate, deviation))

            except (ValueError, IndexError):
                continue

    # 均衡处理：使最终数量相等
    if len(below_data) != len(above_data):
        # 按偏离程度排序（优先保留偏离大的）
        below_data.sort(key=lambda x: x[3], reverse=True)  # 偏离程度降序
        above_data.sort(key=lambda x: x[3], reverse=True)  # 偏离程度降序
        
        min_size = min(len(below_data), len(above_data))
        below_data = below_data[:min_size]  # 保留偏离大的条目
        above_data = above_data[:min_size]  # 保留偏离大的条目

    # 提取最终的层头对
    below_list = [(item[0], item[1]) for item in below_data]
    above_list = [(item[0], item[1]) for item in above_data]

    coverage_data = {
        "below_size": len(below_list),
        "below_pairs": below_list,
        "above_size": len(above_list),
        "above_pairs": above_list,
    }

    try:
        with open(output_file, 'w') as f:
            json.dump(coverage_data, f, indent=4)
        abs_path = os.path.abspath(output_file)
        print(f"分析结果已保存到: {abs_path}")
        print(f"文件大小: {os.path.getsize(output_file)/1024:.2f} KB")
        print(f"全局平均覆盖率: {global_avg_rate:.4f}")
        print(f"调整阈值: 低于 <{below_threshold:.4f}, 高于 >{above_threshold:.4f}")
        print(f"均衡后的低于阈值数量: {len(below_list)}")
        print(f"均衡后的高于阈值数量: {len(above_list)}")
    except Exception as e:
        print(f"保存文件时出错: {e}")


if __name__ == "__main__":
    coverage_file = "/home/v-shumingguo/gsm_blob/results/profile/Qwen3-14B/livecodebench_6k_cover_rate.txt"
    output_file = "coverage_profile_livecodebench_10.json"

    analyze_and_save_coverage(coverage_file, output_file)