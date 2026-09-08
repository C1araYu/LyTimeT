#!/usr/bin/env python3
"""
创建数据分割文件 - 基于double_pendulum的分割策略
"""

import json
import os

def create_data_split(num_seq=1200):
    """创建数据分割，按照8:1:1的比例"""
    train_ratio = 0.8
    val_ratio = 0.1
    test_ratio = 0.1
    
    train_end = int(num_seq * train_ratio)      # 0 to 959
    val_end = train_end + int(num_seq * val_ratio)  # 960 to 1079
    
    train_ids = list(range(0, train_end))          # 0-959 (960 sequences)
    val_ids = list(range(train_end, val_end))      # 960-1079 (120 sequences) 
    test_ids = list(range(val_end, num_seq))       # 1080-1199 (120 sequences)
    
    data_split = {
        "train": train_ids,
        "val": val_ids,
        "test": test_ids
    }
    
    print(f"Train: {len(train_ids)} sequences (0-{train_end-1})")
    print(f"Val: {len(val_ids)} sequences ({train_end}-{val_end-1})")
    print(f"Test: {len(test_ids)} sequences ({val_end}-{num_seq-1})")
    
    return data_split

def create_splits_for_dataset(dataset_name):
    """为指定数据集创建分割文件"""
    print(f"\n=== Creating data splits for {dataset_name} ===")
    
    # 创建datainfo目录
    datainfo_dir = f"/home/kuaiyu/tide/datainfo/{dataset_name}"
    os.makedirs(datainfo_dir, exist_ok=True)
    
    # 为不同的seeds创建分割文件
    for seed in [1, 2, 3]:
        data_split = create_data_split()
        split_file = os.path.join(datainfo_dir, f"data_split_dict_{seed}.json")
        
        with open(split_file, "w") as f:
            json.dump(data_split, f, indent=2)
        
        print(f"Created: {split_file}")

def main():
    # 8阶动力学系统列表
    datasets = [
        "coupled_oscillators",  # 4个耦合谐振子
        "lorenz_4d",           # 耦合Lorenz系统
        "rossler_lorenz"       # Rossler-Lorenz耦合系统
    ]
    
    for dataset in datasets:
        create_splits_for_dataset(dataset)
    
    print("\n=== All data splits created successfully! ===")

if __name__ == "__main__":
    main()
