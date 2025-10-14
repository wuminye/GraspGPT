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
from collections import defaultdict
from itertools import permutations

import open3d as o3d


# 导入必要的模块
try:
    from graspGPT.model.precomputed_dataset import PrecomputedDataset
    from graspGPT.model.token_manager import get_token_manager, decode_sequence, encode_sequence
    from graspGPT.model.parser_and_serializer import (
        Parser,
        Serializer,
        Seq,
        Scene,
        SB,
        UNSEG,
        AMODAL,
        GRASP,
        parse_with_cpp,
    )
    from blender.process_grasp import interior_points_to_gripper_params, Grasp
    from graspGPT.model.core import generate_amodal_sequence
except ImportError:
    print("Warning: 无法导入graspGPT模块，请确保在正确的环境中运行")
    import sys
    sys.exit(1)

DEFAULT_GRASP_HEIGHT = 0.02
COORD_TRANSFORM = np.array([
    [-1, 0, 0, 0],
    [0, -1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
], dtype=np.float32)
COORD_TRANSFORM_INV = np.linalg.inv(COORD_TRANSFORM)

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
    
    #print(f"已保存点云: {filename} ({len(points)} 个点)")

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
        #parser = Parser(decoded_tokens)
        #seq = parser.parse()

        seq = parse_with_cpp(decoded_tokens)
        
        print(f"解析成功，序列包含 {len(seq.items)} 个项目")
        return seq
        
    except Exception as e:
        print(f"解析tokens时出错: {e}")
        # 返回空序列
        return Seq([])

def _cbs_to_world(cbs: List, bbox_min: np.ndarray, voxel_size: float) -> np.ndarray:
    """将CB列表中的体素坐标转换为世界坐标数组。"""
    coords = [cb.coord for cb in cbs if cb.coord is not None]
    if not coords:
        return np.empty((0, 3), dtype=np.float32)
    coords_array = np.asarray(coords, dtype=np.float32)
    return coords_array * voxel_size + bbox_min


def extract_pointclouds_by_category(
    seq: Seq,
    bbox_min: np.ndarray,
    voxel_size: float,
) -> Tuple[
    Dict[str, List[np.ndarray]],
    Dict[str, List[np.ndarray]],
    Dict[str, List[np.ndarray]],
    Dict[str, List[np.ndarray]],
]:
    """一次遍历序列，按语法类别收集点云。"""

    scene_clouds: Dict[str, List[np.ndarray]] = defaultdict(list)
    amodal_clouds: Dict[str, List[np.ndarray]] = defaultdict(list)
    unseg_clouds: Dict[str, List[np.ndarray]] = defaultdict(list)
    grasp_clouds: Dict[str, List[np.ndarray]] = defaultdict(list)

    for item in seq.items:
        if isinstance(item, Scene):
            for sb in item.sbs:
                points = _cbs_to_world(sb.cbs, bbox_min, voxel_size)
                if len(points):
                    scene_clouds[sb.tag].append(points)
                    print(f"Scene SB '{sb.tag}': {len(points)} 个点")
        elif isinstance(item, SB):
            points = _cbs_to_world(item.cbs, bbox_min, voxel_size)
            if len(points):
                scene_clouds[item.tag].append(points)
                print(f"Scene SB '{item.tag}': {len(points)} 个点")

        elif isinstance(item, AMODAL):
            points = _cbs_to_world(item.sb.cbs, bbox_min, voxel_size)
            if len(points):
                amodal_clouds[item.sb.tag].append(points)
                print(f"Amodal SB '{item.sb.tag}': {len(points)} 个点")

        elif isinstance(item, UNSEG):
            for sb in item.sbs:
                points = _cbs_to_world(sb.cbs, bbox_min, voxel_size)
                if len(points):
                    unseg_clouds[sb.tag].append(points)
                    print(f"Unseg SB '{sb.tag}': {len(points)} 个点")

        elif isinstance(item, GRASP):
            for gb in item.gbs:
                points = _cbs_to_world(gb.cbs, bbox_min, voxel_size)
                if len(points):
                    grasp_clouds[gb.tag].append(points)
                    #print(f"GRASP '{gb.tag}': {len(points)} 个点")

    return scene_clouds, amodal_clouds, unseg_clouds, grasp_clouds

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


def reconstruct_grasp_from_points(points: np.ndarray) -> Optional[Grasp]:
    """尝试从抓取点云的3个内点还原一个Grasp对象。"""
    points = np.asarray(points, dtype=np.float32)
    if points.shape != (3, 3):
        return None

    # 将点转换回grasp参数坐标系
    homogeneous = np.hstack([points, np.ones((points.shape[0], 1), dtype=points.dtype)])
    transformed = (COORD_TRANSFORM_INV @ homogeneous.T).T[:, :3]

    left, right, bottom = transformed

    best_candidate: Optional[Tuple[np.ndarray, np.ndarray, float, float]] = None
    best_cost = np.inf

    for order in permutations(range(3)):

        left, right, bottom = transformed[list(order)]

        '''
        line_vec = right - left
        lr_dist = np.linalg.norm(line_vec)
        if lr_dist < 1e-6:
            continue
        line_unit = line_vec / lr_dist

        bottom_vec = bottom - left
        bottom_proj = np.dot(bottom_vec, line_unit)
        bottom_to_line_vec = bottom_vec - bottom_proj * line_unit
        bottom_line_dist = np.linalg.norm(bottom_to_line_vec)

        proj_left = np.dot(left - bottom, line_unit)
        proj_right = np.dot(right - bottom, line_unit)
        if abs(proj_left) <= abs(proj_right):
            continue

        cost = bottom_line_dist
        mid_proj = lr_dist / 2.0
        cost += abs(bottom_proj - mid_proj)
        if not (0.0 <= bottom_proj <= lr_dist):
            cost += 0.05
        '''

        try:
            candidate = interior_points_to_gripper_params(left, right, bottom)
        except Exception:
            continue
        best_candidate = candidate

        center, rotation, width, depth = candidate

        break
        '''
        if not np.isfinite(width) or not np.isfinite(depth) or width <= 0 or depth <= 0:
            continue

        if cost < best_cost:
            best_cost = cost
            best_candidate = candidate
        '''

    if best_candidate is None:
        for order in permutations(range(3)):
            left, right, bottom = transformed[list(order)]
            try:
                candidate = interior_points_to_gripper_params(left, right, bottom)
            except Exception:
                continue
            center, rotation, width, depth = candidate
            if np.isfinite(width) and np.isfinite(depth) and width > 0 and depth > 0:
                best_candidate = candidate
                break
        if best_candidate is None:
            return None

    center, rotation, width, depth = best_candidate

    center_h = np.append(center, 1.0)
    center_world = (COORD_TRANSFORM @ center_h)[:3]
    rotation_world = COORD_TRANSFORM[:3, :3] @ rotation

    grasp = Grasp()
    grasp.score = 1.0
    grasp.width = float(width)
    grasp.height = DEFAULT_GRASP_HEIGHT
    grasp.depth = float(depth)
    grasp.rotation_matrix = rotation_world
    grasp.translation = center_world
    return grasp

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
        dataset = PrecomputedDataset(dataset_path, max_sequence_length=16800, real_filter_mode='amodal_only',apply_del_amodal_sequence=True)
        print(f"Dataset加载成功，包含 {len(dataset)} 个样本")
    except Exception as e:
        print(f"加载dataset失败: {e}")
        return
    
    if sample_idx >= len(dataset):
        print(f"样本索引 {sample_idx} 超出范围，最大索引为 {len(dataset)-1}")
        return
    
    print(f"正在提取样本 {sample_idx}...")
    
    # 获取样本
    tokens, max_seq_len, _ = dataset[sample_idx]

    

    print(f"Tokens shape: {tokens.shape}")
    print(f"Max sequence length: {max_seq_len}")

    # 获取体素信息
    volume_dims = dataset.volume_dims
    bbox_min = dataset.bbox_min
    voxel_size = dataset.voxel_size
    
    print(f"体素信息: dims={volume_dims}, bbox_min={bbox_min}, voxel_size={voxel_size}")


    decoded_tokens = decode_sequence(tokens.squeeze().numpy(), dataset.token_mapping)
    
    #decoded_tokens = generate_amodal_sequence(decoded_tokens, voxel_dims=(volume_dims[0], volume_dims[1], volume_dims[2]))
    tokens = encode_sequence(decoded_tokens, dataset.token_mapping)

    # 创建输出目录
    sample_dir = Path(output_dir) / f"sample_{sample_idx}"
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    # 使用visualize_tokens函数来保存点云
    print("使用visualize_tokens保存点云...")
    visualize_tokens(
        tokens=tokens,
        token_mapping=dataset.token_mapping,
        volume_dims=volume_dims,
        bbox_min=bbox_min,
        voxel_size=voxel_size,
        output_dir=str(sample_dir)
    )
    
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
    
    print("正在按类别提取点云...")
    scene_clouds, amodal_clouds, unseg_clouds, grasp_clouds = extract_pointclouds_by_category(
        seq, bbox_min, voxel_size
    )

    def save_category(category: str, clouds: Dict[str, List[np.ndarray]], brightness_scale: float = 1.0) -> None:
        category_dir = output_path / category
        category_dir.mkdir(parents=True, exist_ok=True)

        merged_points: List[np.ndarray] = []
        merged_colors: List[np.ndarray] = []

        for tag, point_groups in clouds.items():
            for idx, points in enumerate(point_groups):
                if len(points) == 0:
                    continue
                colors = generate_colors_for_object(tag, len(points))
                if brightness_scale != 1.0:
                    colors = np.clip(colors * brightness_scale, 0, 255).astype(np.uint8)

                suffix = f"_{idx}" if idx > 0 else ""
                filename = category_dir / f"{tag}{suffix}.ply"
                save_pointcloud_as_ply(points, str(filename), colors)

                merged_points.append(points)
                merged_colors.append(colors)

        if merged_points:
            merged_points_arr = np.vstack(merged_points)
            merged_colors_arr = np.vstack(merged_colors)
            merged_name = category_dir / f"merged_{category}.ply"
            save_pointcloud_as_ply(merged_points_arr, str(merged_name), merged_colors_arr)
            print(f"合并{category}点云: {len(merged_points_arr)} 个点")
        else:
            print(f"{category} 类别没有点云")

    save_category("scene", scene_clouds)
    save_category("amodal", amodal_clouds)
    save_category("unseg", unseg_clouds)
    save_category("grasp", grasp_clouds, brightness_scale=1.5)

    grasp_mesh_dir = output_path / "grasp_meshes"
    grasp_mesh_dir.mkdir(parents=True, exist_ok=True)
    mesh_count = 0

    for tag, grasp_list in grasp_clouds.items():
        for idx, points in enumerate(grasp_list):
            grasp = reconstruct_grasp_from_points(points)
            if grasp is None:
                print(f"Warning: 无法从抓取点云重建mesh (tag={tag}, index={idx})，跳过保存。")
                continue

            mesh = grasp.to_open3d_geometry()
            mesh_filename = grasp_mesh_dir / f"grasp_mesh_{tag}_{idx}.ply"
            o3d.io.write_triangle_mesh(str(mesh_filename), mesh, write_ascii=False, compressed=False, print_progress=False)
            mesh_count += 1

    if mesh_count > 0:
        print(f"已生成并保存 {mesh_count} 个抓取mesh，目录: {grasp_mesh_dir}")
    else:
        print("未生成任何抓取mesh。")

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
