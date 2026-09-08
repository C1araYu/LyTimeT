import json
import os

# 获取实际存在的序列ID
data_dir = "/home/kuaiyu/tide/data/physics_prediction_v2/data/swing_stick"
existing_ids = []

# 检查0-84范围内的序列
for i in range(85):  # 0-84
    seq_path = os.path.join(data_dir, str(i))
    if os.path.exists(seq_path):
        existing_ids.append(i)

print(f"Found {len(existing_ids)} existing sequences: {existing_ids}")

# 创建新的数据分割 (大约 70% 训练, 15% 验证, 15% 测试)
total = len(existing_ids)
train_end = int(total * 0.7)
val_end = train_end + int(total * 0.15)

train_ids = existing_ids[:train_end]
val_ids = existing_ids[train_end:val_end]
test_ids = existing_ids[val_end:]

new_split = {
    "train": train_ids,
    "val": val_ids,
    "test": test_ids
}

print(f"Train: {len(train_ids)} sequences")
print(f"Val: {len(val_ids)} sequences") 
print(f"Test: {len(test_ids)} sequences")

# 创建目录并保存新的分割文件
os.makedirs("/home/kuaiyu/tide/datainfo/swing_stick", exist_ok=True)

# 为不同的seed创建分割文件
for seed in [1, 2, 3]:
    with open(f"/home/kuaiyu/tide/datainfo/swing_stick/data_split_dict_{seed}.json", "w") as f:
        json.dump(new_split, f, indent=2)
    print(f"Created data_split_dict_{seed}.json")

print("All data split files created successfully!")
