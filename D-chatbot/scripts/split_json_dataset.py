import json
import os
from math import ceil

# 配置
input_file = "data/cleaned/strategy1.json"  # 原始 JSON 文件路径
output_dir = "data/flited"         # 输出文件夹
items_per_file = 300                # 每个文件包含的 JSON 条数

# 创建输出目录（如果不存在）
os.makedirs(output_dir, exist_ok=True)

# 读取原始 JSON
with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

total_items = len(data)
num_files = ceil(total_items / items_per_file)

for i in range(num_files):
    start_idx = i * items_per_file
    end_idx = min(start_idx + items_per_file, total_items)
    chunk = data[start_idx:end_idx]

    # 文件名 data01.json, data02.json ...
    file_name = f"data{str(i+1).zfill(2)}.json"
    file_path = os.path.join(output_dir, file_name)

    with open(file_path, "w", encoding="utf-8") as out_f:
        json.dump(chunk, out_f, ensure_ascii=False, indent=2)

    print(f"Saved {len(chunk)} items to {file_path}")

print(f"Total {num_files} files created, each up to {items_per_file} items.")
