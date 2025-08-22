import open3d as o3d
import os
import numpy as np
from pathlib import Path

def simplify_mesh_file(input_path, output_path, target_triangles=1000):
    """
    使用Open3D简化单个PLY文件的mesh
    
    Parameters:
    - input_path: 输入PLY文件路径
    - output_path: 输出简化PLY文件路径
    - target_triangles: 目标三角形数量
    """
    try:
        # 加载mesh
        mesh = o3d.io.read_triangle_mesh(input_path)
        
        if len(mesh.vertices) == 0:
            print(f"警告: {input_path} 无法加载或为空mesh")
            return False
            
        original_triangles = len(mesh.triangles)
        original_vertices = len(mesh.vertices)
        print(f"原始mesh: {original_vertices} 顶点, {original_triangles} 三角形")
        
        # 如果原始三角形数量已经小于目标数量，直接复制文件
        if original_triangles <= target_triangles:
            print(f"原始mesh已经足够简单，直接保存")
            o3d.io.write_triangle_mesh(output_path, mesh)
            return True
        
        # 使用四边形坍缩算法简化mesh
        simplified_mesh = mesh.simplify_quadric_decimation(target_triangles)
        
        # 移除孤立顶点和三角形
        simplified_mesh.remove_degenerate_triangles()
        simplified_mesh.remove_duplicated_triangles()
        simplified_mesh.remove_duplicated_vertices()
        simplified_mesh.remove_non_manifold_edges()
        
        final_triangles = len(simplified_mesh.triangles)
        final_vertices = len(simplified_mesh.vertices)
        
        print(f"简化后mesh: {final_vertices} 顶点, {final_triangles} 三角形")
        print(f"简化率: {final_triangles/original_triangles:.2%}")
        
        # 保存简化后的mesh
        success = o3d.io.write_triangle_mesh(output_path, simplified_mesh)
        
        if success:
            print(f"成功保存到: {output_path}")
            return True
        else:
            print(f"保存失败: {output_path}")
            return False
            
    except Exception as e:
        print(f"处理 {input_path} 时出错: {str(e)}")
        return False

def simplify_ply_files(ply_files_folder, target_triangles=10000, output_suffix="_simplified"):
    """
    批量简化PLY文件夹中的所有nontextured_simplified.ply文件
    
    Parameters:
    - ply_files_folder: 包含编号子目录的基础文件夹
    - target_triangles: 目标三角形数量
    - output_suffix: 输出文件后缀
    """
    
    # 获取所有PLY文件路径
    ply_files = []
    folder_path = Path(ply_files_folder)
    
    for i in range(88):  # 基于原脚本的范围
        ply_path = folder_path / f'{i:03d}' / 'nontextured_simplified.ply'
        if ply_path.exists():
            ply_files.append(ply_path)
    
    if not ply_files:
        print("未找到PLY文件！")
        return
    
    print(f"找到 {len(ply_files)} 个PLY文件需要简化")
    
    success_count = 0
    
    for i, ply_file in enumerate(ply_files):
        print(f"\n处理 {i+1}/{len(ply_files)}: {ply_file}")
        
        # 生成输出路径
        output_path = ply_file.parent / f"nontextured_simplified{output_suffix}.ply"
        
        # 简化mesh
        if simplify_mesh_file(str(ply_file), str(output_path), target_triangles):
            success_count += 1
    
    print(f"\n简化完成！成功处理 {success_count}/{len(ply_files)} 个文件")

def simplify_single_ply(input_file, output_file=None, target_triangles=1000):
    """
    简化单个PLY文件
    
    Parameters:
    - input_file: 输入PLY文件路径
    - output_file: 输出文件路径（如果为None，则在原文件名后加_simplified）
    - target_triangles: 目标三角形数量
    """
    if output_file is None:
        input_path = Path(input_file)
        output_file = input_path.parent / f"{input_path.stem}_simplified{input_path.suffix}"
    
    return simplify_mesh_file(input_file, output_file, target_triangles)

# 使用示例
if __name__ == "__main__":
    """
    使用说明：
    1. 安装依赖: pip install open3d
    2. 修改PLY_FILES_FOLDER为你的模型文件夹路径
    3. 调整TARGET_TRIANGLES来控制简化程度（数值越小简化越多）
    4. 运行脚本: python simplify_mesh.py
    
    输出：
    - 简化后的PLY文件会保存在各自的目录下
    - 文件名格式: nontextured_simplified_simplified.ply
    """
    
    # 配置参数
    PLY_FILES_FOLDER = "data/models"  # 替换为你的PLY文件路径
    TARGET_TRIANGLES = 4000  # 目标三角形数量（数值越小简化越多）
    OUTPUT_SUFFIX = "_blender_3"  # 输出文件后缀
    
    # 执行批量简化
    simplify_ply_files(
        ply_files_folder=PLY_FILES_FOLDER,
        target_triangles=TARGET_TRIANGLES,
        output_suffix=OUTPUT_SUFFIX
    )
    
    # 单文件简化示例（可选）
    # simplify_single_ply("path/to/your/file.ply", target_triangles=500)