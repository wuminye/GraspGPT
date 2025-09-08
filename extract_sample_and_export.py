#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
功能：从dataset中取一个样本，将其token ids 转换成sequence 序列，
然后将场景中的SB 和grasp变成点云ply文件存储下来

作者：Claude Code
"""

import torch
import numpy as np
import os
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# 导入必要的模块
try:
    from graspGPT.model.precomputed_dataset import PrecomputedDataset
    from graspGPT.model.token_manager import get_token_manager, decode_sequence
    from graspGPT.model.parser_and_serializer import Parser, Serializer, Seq, SB, GRASP
except ImportError:
    print("Warning: 无法导入graspGPT模块，请确保在正确的环境中运行")
    import sys
    sys.exit(1)

def save_pointcloud_as_ply(points: np.ndarray, filename: str, colors: Optional[np.ndarray] = None):
    """
    保存点云为PLY格式文件
    
    Args:
        points: N×3的点云坐标数组
        filename: 输出文件名
        colors: N×3的颜色数组（可选，RGB值0-255）
    """
    if len(points) == 0:
        print(f"Warning: 空点云，跳过保存 {filename}")
        return
        
    # 确保是numpy数组
    points = np.array(points)
    if points.shape[1] != 3:
        print(f"Warning: 点云维度不正确 {points.shape}，跳过保存 {filename}")
        return
    
    # 创建输出目录
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    with open(filename, 'w') as f:
        # PLY文件头
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        
        if colors is not None:
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
        
        f.write("end_header\n")
        
        # 写入点云数据
        for i, point in enumerate(points):
            if colors is not None and i < len(colors):
                color = colors[i]
                f.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} {int(color[0])} {int(color[1])} {int(color[2])}\n")
            else:
                f.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f}\n")
    
    print(f"已保存点云: {filename} ({len(points)} 个点)")

def tokens_ids_to_sequence(token_ids: List[int], token_mapping: Dict) -> Seq:
    """
    将token ids转换为序列
    
    Args:
        token_ids: token id列表
        token_mapping: token到id的映射字典
        
    Returns:
        Seq: 解析后的序列对象
    """
    # 创建反向映射
    inv_mapping = {v: k for k, v in token_mapping.items()}
    
    # 解码token ids为tokens
    try:
        decoded_tokens = decode_sequence(token_ids, token_mapping)
        print(f"解码得到的tokens数量: {len(decoded_tokens)}")
        print(f"前10个tokens: {decoded_tokens[:10] if len(decoded_tokens) > 10 else decoded_tokens}")
        
        # 使用parser解析tokens为AST
        parser = Parser(decoded_tokens)
        seq = parser.parse()
        
        print(f"解析成功，序列包含 {len(seq.items)} 个项目")
        return seq
        
    except Exception as e:
        print(f"解析tokens时出错: {e}")
        # 返回空序列
        return Seq([])

def extract_sb_pointclouds(seq: Seq, volume_dims: Tuple[int, int, int], 
                          bbox_min: np.ndarray, voxel_size: float) -> Dict[str, np.ndarray]:
    """
    从序列中提取SB点云
    
    Args:
        seq: 解析后的序列
        volume_dims: 体素维度
        bbox_min: 边界框最小坐标
        voxel_size: 体素大小
        
    Returns:
        Dict[str, np.ndarray]: 标签到点云的映射
    """
    sb_clouds = {}
    
    for item in seq.items:
        if isinstance(item, SB):
            # 收集该SB的所有坐标
            print(f"Processing SB with tag: {item.tag}, number of CBs: {len(item.cbs)}")
            coords = []
            for cb in item.cbs:
                coords.append(cb.coord)
            
            if coords:
                # 转换体素坐标到世界坐标
                coords_array = np.array(coords)
                # 体素坐标转换为实际坐标
                real_coords = coords_array * voxel_size + bbox_min
                sb_clouds[item.tag] = real_coords
                print(f"SB '{item.tag}': {len(coords)} 个点")
    
    return sb_clouds

def extract_grasp_pointclouds(seq: Seq, volume_dims: Tuple[int, int, int], 
                             bbox_min: np.ndarray, voxel_size: float) -> Dict[str, List[np.ndarray]]:
    """
    从序列中提取GRASP点云
    
    Args:
        seq: 解析后的序列
        volume_dims: 体素维度
        bbox_min: 边界框最小坐标
        voxel_size: 体素大小
        
    Returns:
        Dict[str, List[np.ndarray]]: 标签到抓取点云列表的映射
    """
    grasp_clouds = {}
    
    for item in seq.items:
        if isinstance(item, GRASP):
            for gb in item.gbs:
                if gb.tag not in grasp_clouds:
                    grasp_clouds[gb.tag] = []
                
                # 收集该GB的所有坐标
                coords = []
                for cb in gb.cbs:
                    coords.append(cb.coord)
                
                if coords:
                    # 转换体素坐标到世界坐标
                    coords_array = np.array(coords)
                    real_coords = coords_array * voxel_size + bbox_min
                    grasp_clouds[gb.tag].append(real_coords)
                    print(f"GRASP '{gb.tag}': {len(coords)} 个点")
    
    return grasp_clouds

def generate_colors_for_object(tag: str, num_points: int) -> np.ndarray:
    """
    为对象生成颜色
    
    Args:
        tag: 对象标签
        num_points: 点数量
        
    Returns:
        np.ndarray: N×3的颜色数组
    """
    # 根据标签生成固定颜色
    if tag.startswith('object'):
        try:
            obj_id = int(tag[6:])  # 提取object后面的数字
            # 使用简单的颜色映射
            colors = [
                [255, 0, 0],    # 红色
                [0, 255, 0],    # 绿色  
                [0, 0, 255],    # 蓝色
                [255, 255, 0],  # 黄色
                [255, 0, 255],  # 品红
                [0, 255, 255],  # 青色
                [128, 128, 128] # 灰色
            ]
            color = colors[obj_id % len(colors)]
        except ValueError:
            color = [128, 128, 128]  # 默认灰色
    else:
        color = [128, 128, 128]  # unknow等其他标签用灰色
    
    return np.array([color] * num_points)

def process_dataset_sample(dataset_path: str, sample_idx: int, output_dir: str):
    """
    处理dataset中的一个样本
    
    Args:
        dataset_path: dataset路径
        sample_idx: 样本索引
        output_dir: 输出目录
    """
    print(f"正在加载dataset: {dataset_path}")
    
    # 加载dataset
    try:
        dataset = PrecomputedDataset(dataset_path, max_sequence_length=4096)
        print(f"Dataset加载成功，包含 {len(dataset)} 个样本")
    except Exception as e:
        print(f"加载dataset失败: {e}")
        return
    
    if sample_idx >= len(dataset):
        print(f"样本索引 {sample_idx} 超出范围，最大索引为 {len(dataset)-1}")
        return
    
    print(f"正在提取样本 {sample_idx}...")
    
    # 获取样本
    tokens, max_seq_len, scene_grasps = dataset[sample_idx]

   

    print(f"Tokens shape: {tokens.shape}")
    print(f"Max sequence length: {max_seq_len}")
    print(f"Scene grasps: {len(scene_grasps)} 个对象")


    
    # 转换token ids到序列
    print("正在转换token ids到序列...")
    token_ids = tokens.flatten().tolist() if tokens.dim() > 1 else tokens.tolist()
    seq = tokens_ids_to_sequence(token_ids, dataset.token_mapping)
    
    # 获取体素信息
    volume_dims = dataset.volume_dims
    bbox_min = dataset.bbox_min
    voxel_size = dataset.voxel_size
    
    print(f"体素信息: dims={volume_dims}, bbox_min={bbox_min}, voxel_size={voxel_size}")
    
    # 创建输出目录
    sample_dir = Path(output_dir) / f"sample_{sample_idx}"
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存原始序列信息
    with open(sample_dir / "sequence_info.txt", 'w', encoding='utf-8') as f:
        f.write(f"Sample Index: {sample_idx}\n")
        f.write(f"Token IDs Count: {len(token_ids)}\n")
        f.write(f"Sequence Items: {len(seq.items)}\n")
        f.write(f"Volume Dims: {volume_dims}\n")
        f.write(f"Bbox Min: {bbox_min}\n")
        f.write(f"Voxel Size: {voxel_size}\n\n")
        f.write("Parsed Sequence:\n")
        f.write(str(seq))


    
    
    # 提取并保存SB点云
    print("正在提取SB点云...")
    sb_clouds = extract_sb_pointclouds(seq, volume_dims, bbox_min, voxel_size)
    
    for tag, points in sb_clouds.items():
        colors = generate_colors_for_object(tag, len(points))
        filename = sample_dir / f"sb_{tag}.ply"
        save_pointcloud_as_ply(points, str(filename), colors)
    
    # 提取并保存GRASP点云
    print("正在提取GRASP点云...")
    grasp_clouds = extract_grasp_pointclouds(seq, volume_dims, bbox_min, voxel_size)
    
    for tag, grasp_list in grasp_clouds.items():
        for i, points in enumerate(grasp_list):
            # 为抓取点云使用不同的颜色（更亮的颜色）
            colors = generate_colors_for_object(tag, len(points))
            colors = np.minimum(colors * 1.5, 255).astype(np.uint8)  # 增加亮度
            filename = sample_dir / f"grasp_{tag}_{i}.ply"
            save_pointcloud_as_ply(points, str(filename), colors)
    
   
    
    print(f"样本 {sample_idx} 处理完成，结果保存在: {sample_dir}")

def visualize_tokens(tokens, token_mapping: Dict, volume_dims: Tuple[int, int, int], 
                    bbox_min: np.ndarray, voxel_size: float, output_dir: str = "./output/tokens_visual"):
    """
    可视化tokens的函数，将tokens转换为点云并保存
    
    Args:
        tokens: token张量或列表
        token_mapping: token到id的映射字典
        volume_dims: 体素维度
        bbox_min: 边界框最小坐标
        voxel_size: 体素大小  
        output_dir: 输出目录
    """
    print("=== 开始可视化tokens ===")
    
    # 转换token ids到序列
    print("正在转换token ids到序列...")
    if hasattr(tokens, 'flatten'):
        token_ids = tokens.flatten().tolist() if tokens.dim() > 1 else tokens.tolist()
    else:
        token_ids = tokens if isinstance(tokens, list) else list(tokens)
    
    seq = tokens_ids_to_sequence(token_ids, token_mapping)
    
    print(f"体素信息: dims={volume_dims}, bbox_min={bbox_min}, voxel_size={voxel_size}")
    
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 保存序列信息
    with open(output_path / "tokens_sequence_info.txt", 'w', encoding='utf-8') as f:
        f.write(f"Token IDs Count: {len(token_ids)}\n")
        f.write(f"Sequence Items: {len(seq.items)}\n")
        f.write(f"Volume Dims: {volume_dims}\n")
        f.write(f"Bbox Min: {bbox_min}\n")
        f.write(f"Voxel Size: {voxel_size}\n\n")
        f.write("Parsed Sequence:\n")
        f.write(str(seq))
    
    # 提取并保存SB点云
    print("正在提取SB点云...")
    sb_clouds = extract_sb_pointclouds(seq, volume_dims, bbox_min, voxel_size)
    
    for tag, points in sb_clouds.items():
        colors = generate_colors_for_object(tag, len(points))
        filename = output_path / f"sb_{tag}.ply"
        save_pointcloud_as_ply(points, str(filename), colors)
    
    # 提取并保存GRASP点云
    print("正在提取GRASP点云...")
    grasp_clouds = extract_grasp_pointclouds(seq, volume_dims, bbox_min, voxel_size)
    
    for tag, grasp_list in grasp_clouds.items():
        for i, points in enumerate(grasp_list):
            # 为抓取点云使用不同的颜色（更亮的颜色）
            colors = generate_colors_for_object(tag, len(points))
            colors = np.minimum(colors * 1.5, 255).astype(np.uint8)  # 增加亮度
            filename = output_path / f"grasp_{tag}_{i}.ply"
            save_pointcloud_as_ply(points, str(filename), colors)
    
    print(f"tokens可视化完成，结果保存在: {output_path}")
    return output_path

def main():
    parser = argparse.ArgumentParser(description='从dataset中提取样本并导出点云')
    parser.add_argument('--dataset_path', default='./output/precomputed_data', help='Dataset路径')
    parser.add_argument('--sample_idx', type=int, default=24, help='样本索引 (默认: 0)')
    parser.add_argument('--output_dir', default='./output/dataset_visual', help='输出目录 (默认: ./output/dataset_visual)')

    args = parser.parse_args()
    
    print("=== 从Dataset提取样本并导出点云 ===")
    print(f"Dataset路径: {args.dataset_path}")
    print(f"样本索引: {args.sample_idx}")
    print(f"输出目录: {args.output_dir}")
    print()
    
    process_dataset_sample(args.dataset_path, args.sample_idx, args.output_dir)

if __name__ == '__main__':
    main()