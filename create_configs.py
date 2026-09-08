#!/usr/bin/env python3
"""
创建所有8阶动力学系统的配置文件
"""

import os
import yaml

def create_config(dataset_name, seed=1, epochs=800):
    """创建配置文件"""
    config = {
        'lr': 0.001,
        'seed': seed,
        'if_cuda': True,
        'gamma': 0.5,
        'log_dir': '../logs',
        'train_batch': 4,
        'val_batch': 4,
        'test_batch': 4,
        'num_workers': 8,
        'model_name': 'vit-tide',
        'data_filepath': '../data/physics_prediction_v2/data',
        'dataset': dataset_name,
        'num_frames': 60,
        'lr_schedule': [20, 50, 100, 300, 600],
        'lda1': 0.1,
        'lda2': 1.0,
        'lda3': 0.0,
        'beta': 1.0,
        'num_gpus': 1,  # 使用1个GPU
        'epochs': epochs
    }
    return config

def create_configs_for_dataset(dataset_name):
    """为指定数据集创建配置文件"""
    print(f"\n=== Creating configs for {dataset_name} ===")
    
    # 创建配置目录
    config_dir = f"/home/kuaiyu/tide/configs/{dataset_name}/vit"
    os.makedirs(config_dir, exist_ok=True)
    
    # 为不同的seeds创建配置文件
    for seed in [1, 2, 3]:
        config = create_config(dataset_name, seed=seed)
        config_file = os.path.join(config_dir, f"config{seed}.yaml")
        
        with open(config_file, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        print(f"Created: {config_file}")

def main():
    # 8阶动力学系统列表
    datasets = [
        "coupled_oscillators",  # 4个耦合谐振子
        "lorenz_4d",           # 耦合Lorenz系统
        "rossler_lorenz"       # Rossler-Lorenz耦合系统
    ]
    
    for dataset in datasets:
        create_configs_for_dataset(dataset)
    
    print("\n=== All configs created successfully! ===")

if __name__ == "__main__":
    main()
