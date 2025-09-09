#!/usr/bin/env python3
"""
测试脚本：加载VoxelDataset并使用core.py的save_voxels保存样本为obj文件
"""

import sys
import os
import torch
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# 导入所需模块
from graspGPT.model.dataset import VoxelDataset
from graspGPT.model.core import save_voxels
from graspGPT.model.token_manager import get_token_manager

def main():
    # 设置数据路径 - 可以是单个文件或目录
    data_path = "output/pointclouds/"  # 使用包含多个batch文件的目录
    # data_path = "output/pointclouds/voxel_data_batch_0.pth"  # 或者单个文件
    
    print("=== 创建VoxelDataset ===")
    try:
        # 创建数据集
        dataset = VoxelDataset(
            data_path=data_path,
            max_sequence_length=3512,
            weights_only=False,
            grasp_data_path = 'data'
        )
        
        print(f"数据集大小: {len(dataset)} 个样本")
        print(f"词汇表大小: {dataset.get_vocab_size()}")
        print(f"体积维度: {dataset.volume_dims}")
        
    except Exception as e:
        print(f"创建数据集失败: {e}")
        return
    
    if len(dataset) == 0:
        print("数据集为空，无法进行测试")
        return
    
    print("\n=== 获取样本信息 ===")
    # 选择第一个样本进行测试
    sample_idx = 56
    sample_info = dataset.get_sample_info(sample_idx)
    print(f"样本 {sample_idx} 信息:")
    for key, value in sample_info.items():
        print(f"  {key}: {value}")
    
    print("\n=== 获取样本数据 ===")
    try:
        # 获取样本数据
        tokens, max_seq_len, scene_grasps= dataset[sample_idx]
        import pdb; pdb.set_trace()
        # 保存grasp坐标为PLY文件
        dataset._save_grasp_coordinates_ply(scene_grasps, f"grasp_coordinates_{sample_idx}.ply")
        print(f"tokens形状: {tokens.shape}")
        print(f"最大序列长度: {max_seq_len}")
        print(f"前10个token: {tokens[:10].flatten().tolist()}")
        
        # 获取原始token序列（用于save_voxels）
        voxel_data = dataset.data[sample_idx]
        token_sequence = dataset.tokenizer_fn(voxel_data)
        print(f"token序列长度: {len(token_sequence)}")
        
    except Exception as e:
        print(f"获取样本数据失败: {e}")
        return
    
    print("\n=== 将tokens转换为token字符串序列 ===")
    try:
        # 获取token管理器和映射
        token_manager = get_token_manager()
        img_h, img_w, img_d = dataset.volume_dims
        token_mapping = token_manager.generate_mapping(img_h, img_w, img_d)
        reverse_mapping = {v: k for k, v in token_mapping.items()}
        
        # 将token IDs转换为token字符串
        token_strings = []
        for token_id in token_sequence:
            if token_id in reverse_mapping:
                token_strings.append(reverse_mapping[token_id])
            else:
                print(f"警告: 未知token ID {token_id}")
                token_strings.append(f"<UNK:{token_id}>")
        
        print(f"Token字符串序列长度: {len(token_strings)}")
        print(f"前20个tokens: {token_strings[:20]}")
        
    except Exception as e:
        print(f"转换token序列失败: {e}")
        return
    
    print("\n=== 使用save_voxels保存为OBJ文件 ===")
    try:
        # 设置输出文件路径
        output_file = f"test_sample_{sample_idx}.ply"
        
        # 使用save_voxels保存
        num_points = save_voxels(token_strings, output_file)
        
        print(f"成功保存PLY文件: {output_file}")
        print(f"保存的点数: {num_points}")

     
            
    except Exception as e:
        print(f"保存OBJ文件失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()