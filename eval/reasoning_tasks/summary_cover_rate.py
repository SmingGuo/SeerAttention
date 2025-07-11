import os
import argparse
from collections import defaultdict

def calculate_all_weighted_averages(token_budget=4096):
    # 数据存储结构
    layer_head_data = defaultdict(dict)  # (layer, head) -> (coverages, weights)
    all_data = []  # 存储所有(cover_rate, weight)用于全局平均
    
    # 遍历所有文件
    for filename in os.listdir("."):
        if filename.startswith("cover_rate_layer") and filename.endswith(".txt"):
            # 修复文件名解析问题
            try:
                # 移除文件扩展名
                basename = filename[:-4]
                parts = basename.split('_')
                
                # 确保我们有足够的组成部分
                if len(parts) < 4:
                    print(f"文件名格式错误: {filename}, 部分不足")
                    continue
                
                # 解析层号
                layer_str = parts[2]  # "layerX"
                if not layer_str.startswith("layer"):
                    print(f"文件名格式错误: {filename}, 未找到层前缀")
                    continue
                layer_idx = int(layer_str[5:])  # 移除"layer"前缀(5个字符)
                
                # 解析头号
                head_str = parts[3]  # "headX"
                if not head_str.startswith("head"):
                    print(f"文件名格式错误: {filename}, 未找到头前缀")
                    continue
                head_idx = int(head_str[4:])  # 移除"head"前缀(4个字符)
                
            except (IndexError, ValueError) as e:
                print(f"文件名格式错误: {filename}, 错误: {e}")
                continue
            
            # 读取文件数据
            coverages = []
            weights = []
            try:
                with open(filename, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) < 2:
                            continue
                        try:
                            coverage = float(parts[0])
                            batch_size = int(parts[1])  # 假设第二个部分是batch size
                            seqlen = int(parts[2])  # 假设第三个部分是sequence length
                            if seqlen > token_budget:
                                coverages.append(coverage)
                                weights.append(batch_size * seqlen)
                                all_data.append((coverage, batch_size * seqlen))
                        except (ValueError, IndexError) as e:
                            print(f"文件 {filename} 行格式错误: {line}, 错误: {e}")
                layer_head_data[layer_idx][head_idx] = (coverages, weights)
            except Exception as e:
                print(f"处理文件 {filename} 时出错: {e}")

    # 结果字典
    results = {
        "per_layer_head": {},   # (layer, head) -> weighted avg
        "per_layer": {},        # layer -> weighted avg
        "per_head": {},         # head -> weighted avg
        "global_avg": None      # 全局加权平均
    }
    
    # 1. 计算每个(层, 头)的加权平均值
    for layer_idx, heads in layer_head_data.items():
        for head_idx, (coverages, weights) in heads.items():
            if weights:  # 避免空列表
                total_weight = sum(weights)
                weighted_avg = sum(c * w for c, w in zip(coverages, weights)) / total_weight
                results["per_layer_head"][(layer_idx, head_idx)] = weighted_avg
    
    # 2. 计算每层(所有头)的加权平均值
    for layer_idx in layer_head_data:
        all_coverages = []
        all_weights = []
        for head_data in layer_head_data[layer_idx].values():
            all_coverages.extend(head_data[0])
            all_weights.extend(head_data[1])
        
        if all_weights:
            total_weight = sum(all_weights)
            weighted_avg = sum(c * w for c, w in zip(all_coverages, all_weights)) / total_weight
            results["per_layer"][layer_idx] = weighted_avg
    
    # 3. 计算每个头(跨所有层)的加权平均值
    # 先按头汇总数据
    head_data_dict = defaultdict(lambda: ([], []))  # head_idx -> (coverages, weights)
    for layer_idx, heads in layer_head_data.items():
        for head_idx, (coverages, weights) in heads.items():
            head_data_dict[head_idx][0].extend(coverages)
            head_data_dict[head_idx][1].extend(weights)
    
    for head_idx, (coverages, weights) in head_data_dict.items():
        if weights:
            total_weight = sum(weights)
            weighted_avg = sum(c * w for c, w in zip(coverages, weights)) / total_weight
            results["per_head"][head_idx] = weighted_avg
    
    # 4. 计算全局加权平均值
    if all_data:
        all_coverages, all_weights = zip(*all_data)
        total_weight = sum(all_weights)
        global_avg = sum(c * w for c, w in all_data) / total_weight
        results["global_avg"] = global_avg
    
    return results

if __name__ == "__main__":
    # 添加命令行参数解析
    parser = argparse.ArgumentParser(description='计算覆盖率加权平均值并输出结果')
    parser.add_argument('--output', type=str, required=True, 
                        help='输出结果文件路径')
    parser.add_argument('--token_budget', type=int, default=4096)
    args = parser.parse_args()
    
    # 计算所有加权平均值
    results = calculate_all_weighted_averages(token_budget=args.token_budget)
    
    # 确保输出目录存在
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 将结果输出到文件
    with open(args.output, "w") as output_file:
        # 1. 输出每个(层, 头)的加权平均值
        output_file.write("每个(层, 头)的加权平均覆盖率:\n")
        for (layer, head), avg in sorted(results["per_layer_head"].items()):
            output_file.write(f"  层 {layer}, 头 {head}: {avg:.4f}\n")
        output_file.write("\n")
        
        # 2. 输出每层的加权平均值
        output_file.write("每层(所有头)的加权平均覆盖率:\n")
        for layer, avg in sorted(results["per_layer"].items()):
            output_file.write(f"  层 {layer}: {avg:.4f}\n")
        output_file.write("\n")
        
        # 3. 输出每个头的加权平均值
        output_file.write("每个头(跨所有层)的加权平均覆盖率:\n")
        for head, avg in sorted(results["per_head"].items()):
            output_file.write(f"  头 {head}: {avg:.4f}\n")
        output_file.write("\n")
        
        # 4. 输出全局加权平均值
        global_avg = results['global_avg'] if results['global_avg'] is not None else 0
        output_file.write(f"全局加权平均覆盖率: {global_avg:.4f}\n")

    print(f"结果已成功输出到文件: {args.output}")