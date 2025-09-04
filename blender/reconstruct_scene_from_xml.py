#!/usr/bin/env python3
"""
从XML文件和单个物体模型重建场景的整体模型
"""

import os
import xml.etree.ElementTree as ET
import numpy as np
import open3d as o3d
import argparse
from pathlib import Path


def parse_transform_xml(xml_file_path):
    """解析XML文件并提取物体变换信息"""
    try:
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
        
        scene_id = root.get('id')
        objects_info = []
        
        for obj_elem in root.findall('object'):
            obj_id = int(obj_elem.get('id'))
            obj_name = obj_elem.get('name')
            
            # 解析变换矩阵
            transform_elem = obj_elem.find('transform')
            transform_matrix = np.eye(4)
            
            if transform_elem is not None:
                matrix_rows = []
                for row_elem in transform_elem.findall('row'):
                    row_values = [float(x) for x in row_elem.text.split()]
                    matrix_rows.append(row_values)
                transform_matrix = np.array(matrix_rows)
            
            # 解析位置
            pos_elem = obj_elem.find('position')
            position = [0, 0, 0]
            if pos_elem is not None:
                position = [
                    float(pos_elem.get('x')),
                    float(pos_elem.get('y')),
                    float(pos_elem.get('z'))
                ]
            
            # 解析旋转
            rot_elem = obj_elem.find('rotation')
            rotation = [0, 0, 0]
            if rot_elem is not None:
                rotation = [
                    float(rot_elem.get('x')),
                    float(rot_elem.get('y')),
                    float(rot_elem.get('z'))
                ]
            
            # 解析缩放
            scale_elem = obj_elem.find('scale')
            scale = [1, 1, 1]
            if scale_elem is not None:
                scale = [
                    float(scale_elem.get('x')),
                    float(scale_elem.get('y')),
                    float(scale_elem.get('z'))
                ]
            
            objects_info.append({
                'id': obj_id,
                'name': obj_name,
                'transform_matrix': transform_matrix,
                'position': position,
                'rotation': rotation,
                'scale': scale
            })
        
        return scene_id, objects_info
    
    except Exception as e:
        print(f"解析XML文件失败 {xml_file_path}: {e}")
        return None, []


def load_object_model(models_folder, obj_id):
    """加载单个物体模型"""
    # 查找对应的PLY文件
    model_path = os.path.join(models_folder, f'{obj_id:03d}', 'nontextured_simplified_blender_3.ply')
    
    if not os.path.exists(model_path):
        # 尝试其他可能的文件名
        alt_paths = [
            os.path.join(models_folder, f'{obj_id:03d}', 'nontextured.ply'),
            os.path.join(models_folder, f'{obj_id:03d}', 'textured.ply'),
            os.path.join(models_folder, f'{obj_id:03d}.ply'),
        ]
        
        for alt_path in alt_paths:
            if os.path.exists(alt_path):
                model_path = alt_path
                break
        else:
            print(f"警告: 找不到物体模型文件 {obj_id:03d}")
            return None
    
    try:
        # 加载PLY文件
        mesh = o3d.io.read_triangle_mesh(model_path)
        if len(mesh.vertices) == 0:
            print(f"警告: 物体模型 {obj_id:03d} 为空")
            return None
        return mesh
    except Exception as e:
        print(f"加载物体模型失败 {model_path}: {e}")
        return None


def get_coordinate_transform(method=0):
    """使用flip_xy变换方法（X和Y都取负）"""
    # X和Y都取负
    return np.array([
        [-1, 0,  0,  0],
        [0, -1,  0,  0], 
        [0,  0,  1,  0],
        [0,  0,  0,  1]
    ])

def apply_transform(mesh, transform_matrix, apply_coordinate_correction=True):
    """对网格应用变换矩阵"""
    if mesh is None:
        return None
    
    # 复制网格以避免修改原始数据
    try:
        # 新版本Open3D
        transformed_mesh = mesh.copy()
    except AttributeError:
        # 旧版本Open3D或CUDA版本
        transformed_mesh = o3d.geometry.TriangleMesh()
        transformed_mesh.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices))
        transformed_mesh.triangles = o3d.utility.Vector3iVector(np.asarray(mesh.triangles))
        if mesh.has_vertex_normals():
            transformed_mesh.vertex_normals = o3d.utility.Vector3dVector(np.asarray(mesh.vertex_normals))
        if mesh.has_vertex_colors():
            transformed_mesh.vertex_colors = o3d.utility.Vector3dVector(np.asarray(mesh.vertex_colors))
    
    # 应用坐标系校正（如果需要）
    if apply_coordinate_correction:
        # 默认使用方法0（不变换），可以通过修改这里的数字来测试其他方法
        coord_correction = get_coordinate_transform(method=0)
        final_transform = np.dot(transform_matrix, coord_correction)
    else:
        final_transform = transform_matrix
    
    # 应用变换矩阵
    transformed_mesh.transform(final_transform)
    
    return transformed_mesh



def reconstruct_scene(xml_file_path, models_folder, output_path=None):
    """重建场景整体模型"""
    print(f"正在处理: {xml_file_path}")
    
    # 解析XML文件
    scene_id, objects_info = parse_transform_xml(xml_file_path)
    if scene_id is None:
        return None
    
    print(f"场景ID: {scene_id}, 包含 {len(objects_info)} 个物体")
    print("使用flip_xy坐标系变换（X和Y都取负）")
    
    # 重建场景
    scene_meshes = []
    
    for obj_info in objects_info:
        obj_id = obj_info['id']
        transform_matrix = obj_info['transform_matrix']
        
        print(f"  处理物体 {obj_id:03d}...")
        
        # 加载物体模型
        mesh = load_object_model(models_folder, obj_id)
        if mesh is None:
            continue
        
        # 应用变换
        try:
            # 新版本Open3D
            transformed_mesh = mesh.copy()
        except AttributeError:
            # 旧版本Open3D或CUDA版本
            transformed_mesh = o3d.geometry.TriangleMesh()
            transformed_mesh.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices))
            transformed_mesh.triangles = o3d.utility.Vector3iVector(np.asarray(mesh.triangles))
            if mesh.has_vertex_normals():
                transformed_mesh.vertex_normals = o3d.utility.Vector3dVector(np.asarray(mesh.vertex_normals))
            if mesh.has_vertex_colors():
                transformed_mesh.vertex_colors = o3d.utility.Vector3dVector(np.asarray(mesh.vertex_colors))
        
        # 应用坐标系变换（flip_xy）
        coord_correction = get_coordinate_transform()
        final_transform = np.dot(coord_correction, transform_matrix)
        transformed_mesh.transform(final_transform)
        
        # 设置颜色（根据物体ID）
        color = np.array([obj_id / 255.0, obj_id / 255.0, obj_id / 255.0])
        transformed_mesh.paint_uniform_color(color)
        
        scene_meshes.append(transformed_mesh)
    
    if not scene_meshes:
        print("警告: 没有成功加载任何物体")
        return None
    
    # 合并所有网格
    print("合并场景网格...")
    combined_mesh = scene_meshes[0]
    for mesh in scene_meshes[1:]:
        combined_mesh += mesh
    
    # 保存结果
    if output_path:
        print(f"保存场景到: {output_path}")
        success = o3d.io.write_triangle_mesh(output_path, combined_mesh)
        if not success:
            print(f"保存失败: {output_path}")
            return None
    
    print(f"场景重建完成: {len(scene_meshes)} 个物体")
    return combined_mesh


def batch_reconstruct_scenes(xml_folder, models_folder, output_folder):
    """批量重建场景"""
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)
    
    # 查找所有XML文件
    xml_files = []
    for file in os.listdir(xml_folder):
        if file.endswith('_transforms.xml'):
            xml_files.append(file)
    
    xml_files.sort()
    print(f"找到 {len(xml_files)} 个XML文件")
    
    success_count = 0
    
    for xml_file in xml_files:
        xml_path = os.path.join(xml_folder, xml_file)
        
        # 提取场景编号
        scene_name = xml_file.replace('_transforms.xml', '')
        output_path = os.path.join(output_folder, f"{scene_name}_reconstructed.ply")
        
        # 重建场景
        try:
            result = reconstruct_scene(xml_path, models_folder, output_path)
            if result is not None:
                success_count += 1
        except Exception as e:
            print(f"重建场景失败 {xml_file}: {e}")
        
        print("-" * 50)
        
    
    print(f"批量重建完成: {success_count}/{len(xml_files)} 个场景成功")


def main():
    parser = argparse.ArgumentParser(description="从XML文件和单个物体模型重建场景")
    parser.add_argument("--xml_path", type=str, help="单个XML文件路径")
    parser.add_argument("--xml_folder", type=str, default="../output/synthetic_meshes", 
                        help="XML文件夹路径（批量处理）")
    parser.add_argument("--models_folder", type=str, default="../data/models",
                        help="物体模型文件夹路径")
    parser.add_argument("--output_folder", type=str, default="../output/reconstructed_scenes",
                        help="输出文件夹路径")
    parser.add_argument("--output_path", type=str, help="单个输出文件路径")
    
    args = parser.parse_args()
    
    if args.xml_path:
        # 处理单个文件
        output_path = args.output_path
        if not output_path:
            # 根据XML文件名生成输出路径
            xml_name = Path(args.xml_path).stem
            scene_name = xml_name.replace('_transforms', '')
            output_path = os.path.join(args.output_folder, f"{scene_name}_reconstructed.ply")
            os.makedirs(args.output_folder, exist_ok=True)
        
        reconstruct_scene(args.xml_path, args.models_folder, output_path)
    else:
        # 批量处理
        batch_reconstruct_scenes(args.xml_folder, args.models_folder, args.output_folder)


if __name__ == "__main__":
    main()