#!/usr/bin/env python3
"""
Coordinate alignment script: Calculate ground plane normal and transform point clouds
to align Z-axis with the ground normal.
"""

import numpy as np
import open3d as o3d
import os
from pathlib import Path
import xml.etree.ElementTree as ET
import torch
from typing import List, Tuple
import random

# Import GraspGPT modules
try:
    from graspGPT.model.token_manager import get_token_manager
    from graspGPT.model.parser_and_serializer import Serializer, Seq, Scene, SB, CB
except ImportError:
    print("Warning: Could not import GraspGPT modules. Sequence conversion will be skipped.")

def read_ply_ground(filepath):
    """Read ground PLY file and calculate normal vector"""
    mesh = o3d.io.read_triangle_mesh(filepath)
    if len(mesh.vertices) == 0:
        # Try reading as point cloud
        pcd = o3d.io.read_point_cloud(filepath)
        if len(pcd.points) == 0:
            raise ValueError(f"Cannot read PLY file: {filepath}")
        
        # If normal information exists, use it directly
        if len(pcd.normals) > 0:
            normals = np.asarray(pcd.normals)
            # Use average normal as ground normal
            ground_normal = np.mean(normals, axis=0)
            ground_normal = ground_normal / np.linalg.norm(ground_normal)
            return ground_normal
        else:
            # If no normal info, calculate best fit plane
            points = np.asarray(pcd.points)
            return fit_plane_normal(points)
    else:
        # If it's a mesh, compute vertex normals
        mesh.compute_vertex_normals()
        if len(mesh.vertex_normals) > 0:
            normals = np.asarray(mesh.vertex_normals)
            ground_normal = np.mean(normals, axis=0)
            ground_normal = ground_normal / np.linalg.norm(ground_normal)
            return ground_normal
        else:
            # Compute triangle normals
            mesh.compute_triangle_normals()
            if len(mesh.triangle_normals) > 0:
                normals = np.asarray(mesh.triangle_normals)
                ground_normal = np.mean(normals, axis=0)
                ground_normal = ground_normal / np.linalg.norm(ground_normal)
                return ground_normal
    
    raise ValueError("Cannot compute ground normal")

def fit_plane_normal(points):
    """Use SVD to fit plane and return normal vector"""
    # Compute centroid
    centroid = np.mean(points, axis=0)
    # Center points
    centered_points = points - centroid
    # SVD decomposition
    U, S, Vt = np.linalg.svd(centered_points)
    # The vector corresponding to smallest singular value is the normal
    normal = Vt[-1, :]
    return normal / np.linalg.norm(normal)

def compute_rotation_matrix(from_vector, to_vector):
    """Compute rotation matrix from from_vector to to_vector"""
    from_vector = from_vector / np.linalg.norm(from_vector)
    to_vector = to_vector / np.linalg.norm(to_vector)
    
    # If vectors are already aligned
    if np.allclose(from_vector, to_vector):
        return np.eye(3)
    
    # If vectors are opposite
    if np.allclose(from_vector, -to_vector):
        # Find a perpendicular vector
        if abs(from_vector[0]) < 0.9:
            perpendicular = np.array([1, 0, 0])
        else:
            perpendicular = np.array([0, 1, 0])
        
        # Use Gram-Schmidt orthogonalization
        perpendicular = perpendicular - np.dot(perpendicular, from_vector) * from_vector
        perpendicular = perpendicular / np.linalg.norm(perpendicular)
        
        # 180 degree rotation matrix
        return 2 * np.outer(perpendicular, perpendicular) - np.eye(3)
    
    # General case using Rodrigues formula
    v = np.cross(from_vector, to_vector)
    s = np.linalg.norm(v)
    c = np.dot(from_vector, to_vector)
    
    vx = np.array([[0, -v[2], v[1]],
                   [v[2], 0, -v[0]],
                   [-v[1], v[0], 0]])
    
    R = np.eye(3) + vx + np.dot(vx, vx) * ((1 - c) / (s * s))
    return R

def load_all_pointclouds(file_paths):
    """Load all point clouds once and return them with file info"""
    loaded_pcds = []
    
    for file_path in file_paths:
        pcd = o3d.io.read_point_cloud(str(file_path))
        if len(pcd.points) > 0:
            original_count = len(pcd.points)
            
            # Get original bounding box before downsampling
            original_bbox_min = np.asarray(pcd.get_axis_aligned_bounding_box().min_bound)
            original_bbox_max = np.asarray(pcd.get_axis_aligned_bounding_box().max_bound)
            
            # Downsample to 1/10 of original points using uniform sampling
            target_count = max(1, original_count // 10)
            downsampled_pcd = pcd.uniform_down_sample(every_k_points=max(1, original_count // target_count))
            
            print(f"Loaded {file_path.name}: {original_count} -> {len(downsampled_pcd.points)} points")
            
            loaded_pcds.append({
                'file_path': file_path,
                'pcd': downsampled_pcd,
                'original_bbox_min': original_bbox_min,
                'original_bbox_max': original_bbox_max
            })
        else:
            print(f"Warning: Empty point cloud file {file_path}")
    
    return loaded_pcds

def compute_global_bbox_from_loaded(loaded_pcds, rotation_matrix):
    """Compute global bounding box by transforming individual bounding boxes from loaded data"""
    global_min = np.array([np.inf, np.inf, np.inf])
    global_max = np.array([-np.inf, -np.inf, -np.inf])
    
    for pcd_data in loaded_pcds:
        min_bound = pcd_data['original_bbox_min']
        max_bound = pcd_data['original_bbox_max']
        
        # Transform all 8 corners of the bounding box
        corners = np.array([
            [min_bound[0], min_bound[1], min_bound[2]],
            [min_bound[0], min_bound[1], max_bound[2]], 
            [min_bound[0], max_bound[1], min_bound[2]],
            [min_bound[0], max_bound[1], max_bound[2]],
            [max_bound[0], min_bound[1], min_bound[2]],
            [max_bound[0], min_bound[1], max_bound[2]],
            [max_bound[0], max_bound[1], min_bound[2]],
            [max_bound[0], max_bound[1], max_bound[2]]
        ])
        
        # Apply rotation to all corners
        transformed_corners = np.dot(corners, rotation_matrix.T)
        
        # Update global bounds
        corner_min = np.min(transformed_corners, axis=0)
        corner_max = np.max(transformed_corners, axis=0)
        
        global_min = np.minimum(global_min, corner_min)
        global_max = np.maximum(global_max, corner_max)
    
    if np.any(np.isinf(global_min)) or np.any(np.isinf(global_max)):
        raise ValueError("No valid point clouds found")
    
    return global_min, global_max

def compute_global_bbox_from_individual(file_paths, rotation_matrix):
    """Compute global bounding box by transforming individual bounding boxes"""
    global_min = np.array([np.inf, np.inf, np.inf])
    global_max = np.array([-np.inf, -np.inf, -np.inf])
    
    for file_path in file_paths:
        pcd = o3d.io.read_point_cloud(str(file_path))
        if len(pcd.points) > 0:
            # Get original bounding box
            bbox = pcd.get_axis_aligned_bounding_box()
            min_bound = np.asarray(bbox.min_bound)
            max_bound = np.asarray(bbox.max_bound)
            
            # Transform all 8 corners of the bounding box
            corners = np.array([
                [min_bound[0], min_bound[1], min_bound[2]],
                [min_bound[0], min_bound[1], max_bound[2]], 
                [min_bound[0], max_bound[1], min_bound[2]],
                [min_bound[0], max_bound[1], max_bound[2]],
                [max_bound[0], min_bound[1], min_bound[2]],
                [max_bound[0], min_bound[1], max_bound[2]],
                [max_bound[0], max_bound[1], min_bound[2]],
                [max_bound[0], max_bound[1], max_bound[2]]
            ])
            
            # Apply rotation to all corners
            transformed_corners = np.dot(corners, rotation_matrix.T)
            
            # Update global bounds
            corner_min = np.min(transformed_corners, axis=0)
            corner_max = np.max(transformed_corners, axis=0)
            
            global_min = np.minimum(global_min, corner_min)
            global_max = np.maximum(global_max, corner_max)
    
    if np.any(np.isinf(global_min)) or np.any(np.isinf(global_max)):
        raise ValueError("No valid point clouds found")
    
    return global_min, global_max

def compute_global_bbox_after_transform(file_paths, rotation_matrix, translation_vector):
    """Compute global bounding box after applying full transformation"""
    global_min = np.array([np.inf, np.inf, np.inf])
    global_max = np.array([-np.inf, -np.inf, -np.inf])
    
    for file_path in file_paths:
        pcd = o3d.io.read_point_cloud(str(file_path))
        if len(pcd.points) > 0:

            
            # Apply full transformation (rotation + translation)
            pcd.rotate(rotation_matrix, center=(0, 0, 0))
            pcd.translate(translation_vector)
            
            
            # Get bounding box after transformation
            bbox = pcd.get_axis_aligned_bounding_box()
            min_bound = np.asarray(bbox.min_bound)
            max_bound = np.asarray(bbox.max_bound)
            
            # Update global bounds
            global_min = np.minimum(global_min, min_bound)
            global_max = np.maximum(global_max, max_bound)
    
    if np.any(np.isinf(global_min)) or np.any(np.isinf(global_max)):
        raise ValueError("No valid point clouds found")
    
    return global_min, global_max

def compute_global_bbox_after_transform_loaded(loaded_pcds, rotation_matrix, translation_vector):
    """Compute global bounding box after applying full transformation using loaded point clouds"""
    global_min = np.array([np.inf, np.inf, np.inf])
    global_max = np.array([-np.inf, -np.inf, -np.inf])
    
    for pcd_data in loaded_pcds:
        # Create a copy to avoid modifying original
        pcd = o3d.geometry.PointCloud(pcd_data['pcd'])
        
        # Apply full transformation (rotation + translation)
        pcd.rotate(rotation_matrix, center=(0, 0, 0))
        pcd.translate(translation_vector)
        
        # Get bounding box after transformation
        bbox = pcd.get_axis_aligned_bounding_box()
        min_bound = np.asarray(bbox.min_bound)
        max_bound = np.asarray(bbox.max_bound)
        
        # Update global bounds
        global_min = np.minimum(global_min, min_bound)
        global_max = np.maximum(global_max, max_bound)
    
    if np.any(np.isinf(global_min)) or np.any(np.isinf(global_max)):
        raise ValueError("No valid point clouds found")
    
    return global_min, global_max

def filter_points_by_bbox(pcd: o3d.geometry.PointCloud, bbox_min: np.ndarray, bbox_max: np.ndarray) -> o3d.geometry.PointCloud:
    """Filter point cloud to keep only points within bbox"""
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors) if len(pcd.colors) > 0 else None
    
    # Find points within bbox
    mask = ((points[:, 0] >= bbox_min[0]) & (points[:, 0] <= bbox_max[0]) &
            (points[:, 1] >= bbox_min[1]) & (points[:, 1] <= bbox_max[1]) &
            (points[:, 2] >= bbox_min[2]) & (points[:, 2] <= bbox_max[2]))
    
    if np.sum(mask) / len(mask) < 0.05:
        print("Warning: 95% or more points are outside the bounding box")
        exit(1)
    
    # Filter points
    filtered_points = points[mask]
    
    # Create new point cloud
    filtered_pcd = o3d.geometry.PointCloud()
    filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)
    
    # Filter colors if they exist
    if colors is not None:
        filtered_colors = colors[mask]
        filtered_pcd.colors = o3d.utility.Vector3dVector(filtered_colors)
    
    removed_count = len(points) - len(filtered_points)
    
    return filtered_pcd

def voxel_downsample_with_colors(pcd, voxel_size=0.01, bbox_min=None, bbox_max=None):
    """Downsample point cloud using voxel grid with color averaging"""
    if len(pcd.points) == 0:
        return pcd, []
    
    # Get points and colors
    points = np.asarray(pcd.points)
    has_colors = len(pcd.colors) > 0
    
    if has_colors:
        colors = np.round(np.asarray(pcd.colors) * 255.0) # Convert to 0-255 range
    else:
        # Use default gray color if no colors
        colors = np.full((len(points), 3), 127.0)  # Gray color
        has_colors = True

    # Use bbox to determine voxel grid origin if provided
    if bbox_min is not None:
        # Calculate voxel indices relative to bbox origin
        voxel_indices = np.floor((points - bbox_min) / voxel_size).astype(int)
        grid_origin = bbox_min
    else:
        # Calculate voxel indices for each point (original method)
        voxel_indices = np.floor(points / voxel_size).astype(int)
        grid_origin = np.zeros(3)
    
    # Create unique voxel dictionary
    voxel_dict = {}
    
    for i, voxel_idx in enumerate(voxel_indices):
        voxel_key = tuple(voxel_idx)
        
        if voxel_key not in voxel_dict:
            voxel_dict[voxel_key] = {
                'points': [],
                'colors': []
            }
        
        voxel_dict[voxel_key]['points'].append(points[i])
        voxel_dict[voxel_key]['colors'].append(colors[i])
    
    # Generate downsampled points and colors
    downsampled_points = []
    downsampled_colors = []
    voxel_coordinates = []  # Store integer voxel coordinates
    
    for voxel_key, voxel_data in voxel_dict.items():
        # Voxel center position
        if bbox_min is not None:
            voxel_center = grid_origin + (np.array(voxel_key) + 0.5) * voxel_size
        else:
            voxel_center = (np.array(voxel_key) + 0.5) * voxel_size
       
        # Most frequent color
        if voxel_data['colors']:
            colors = np.array(voxel_data['colors'])
            # Convert to integers and find most frequent color
            int_colors = colors.astype(int)
            unique_colors, counts = np.unique(int_colors, axis=0, return_counts=True)
            most_frequent_color = unique_colors[np.argmax(counts)]
            most_frequent_color = most_frequent_color.astype(np.float32) / 255.0  # Normalize back to [0, 1]
            downsampled_colors.append(most_frequent_color)
            
            downsampled_points.append(voxel_center)
        
            # Store integer voxel coordinates
            voxel_coordinates.append(voxel_key)

    # Create new point cloud
    downsampled_pcd = o3d.geometry.PointCloud()
    downsampled_pcd.points = o3d.utility.Vector3dVector(np.array(downsampled_points))
    
    if downsampled_colors:
        downsampled_pcd.colors = o3d.utility.Vector3dVector(np.array(downsampled_colors))
    
    return downsampled_pcd, voxel_coordinates

def create_voxel_data_list(voxel_coordinates: List[Tuple], colors: np.ndarray) -> List[Tuple]:
    """Create list of (color_255, coordinates_tensor) tuples from voxel data"""
    # Create data array for sorting
    data_list = []
    for voxel_coord, color in zip(voxel_coordinates, colors):
        # Convert color from [0,1] to [0,255] range
        color_255 = np.round(color * 255.0).astype(int)
        data_list.append({
            'coord': voxel_coord,
            'color_255': color_255[0]
        })
    
    # Group by color
    color_groups = {}
    for data in data_list:
        color_255 = data['color_255']
        coord = data['coord']
        
        if color_255 not in color_groups:
            color_groups[color_255] = []
        color_groups[color_255].append(coord)

    # Sort coordinates within each color group and create result list
    result_list = []
    for color_255 in sorted(color_groups.keys()):
        coords = color_groups[color_255]
        # Sort coordinates by (x, y, z)
        coords.sort(key=lambda x: (x[0], x[1], x[2]))
        # Convert to tensor
        coord_tensor = torch.tensor(coords, dtype=torch.uint8)
        result_list.append((color_255, coord_tensor))
    
    return result_list

def convert_data_list_to_sequence(data_list: List[Tuple], volume_dims: Tuple[int, int, int] = (80, 54, 34)) -> List[int]:
    """
    Convert data_list to token sequence using the same logic as precomputed_dataset.py
    
    Args:
        data_list: List of (color_255, coordinates_tensor) tuples
        volume_dims: Volume dimensions for token mapping
        
    Returns:
        List[int]: List of token IDs representing the sequence
    """
    try:
        # Setup token manager and tokenizer
        token_manager = get_token_manager()
        token_mapping = token_manager.generate_mapping(volume_dims[0], volume_dims[1], volume_dims[2])
        
        # Step 1: Collect all SBs from the data_list
        sbs = []
        
        # Randomly shuffle the data_list order
        data_list_copy = data_list.copy()
        random.shuffle(data_list_copy)
        
        for color, coordinates in data_list_copy:
            # Map color to shape tag - use object tags based on color value
            if 0 <= color <= 87:
                shape_tag = f'object{color:02d}'  # object00 to object87
            else:
                shape_tag = 'unknow'  # fallback for out-of-range colors
            
            # Create coordinate blocks (CB) from coordinates
            cbs = []
            coords_list = coordinates.tolist()
            
            for coord in coords_list:
                x, y, z = coord
                # Ensure coordinates are integers and within bounds
                x, y, z = int(x), int(y), int(z)
                coord_tuple = (x, y, z)
                
                # Create CB with coordinate
                cb = CB(coord=coord_tuple)
                cbs.append(cb)
            
            # Create SB (Segment Block) with the shape tag and coordinate blocks
            sb = SB(tag=shape_tag, cbs=cbs)
            sbs.append(sb)
        
        # Step 2: Create sequence with Scene node wrapping SBs
        scene = Scene(sbs=sbs)
        seq = Seq(items=[scene])
        
        # Step 3: Serialize AST to flat token list
        flat_tokens = Serializer.serialize(seq)
        
        # Step 4: Convert flat tokens to token IDs using token_manager mapping
        token_ids = []
        
        for token in flat_tokens:
            if token in token_mapping:
                token_ids.append(token_mapping[token])
            else:
                # Handle unknown tokens - could add to mapping or use special token
                print(f"Warning: Unknown token '{token}' not in mapping")
                # For robustness, you might want to add a special <UNK> token
                # For now, we'll skip unknown tokens
                continue
        
        return token_ids, seq
        
    except Exception as e:
        print(f"Warning: Could not convert data_list to sequence: {e}")
        return []

def transform_and_save_pointcloud(pcd_data, output_path, rotation_matrix, translation_vector, voxel_size=0.01, bbox_min=None, bbox_max=None):
    """Apply translation and rotation transform to loaded point cloud, downsample, and save"""
    # Create a copy to avoid modifying original
    pcd = o3d.geometry.PointCloud(pcd_data['pcd'])
    
    # Apply rotation transform
    pcd.rotate(rotation_matrix, center=(0, 0, 0))

    # Apply translation transform
    pcd.translate(translation_vector)
    
    # Check if points are within bbox before voxelization
    if bbox_min is not None and bbox_max is not None:
        pcd = filter_points_by_bbox(pcd, bbox_min, bbox_max)
    
    # Apply voxel downsampling with bbox
    downsampled_pcd, voxel_coordinates = voxel_downsample_with_colors(pcd, voxel_size, bbox_min, bbox_max)
    
    print(f"Downsampled from {len(pcd.points)} to {len(downsampled_pcd.points)} points")
    
    # Save transformed and downsampled point cloud
    success = o3d.io.write_point_cloud(output_path, downsampled_pcd)
    if success:
        print(f"Saved transformed point cloud: {output_path}")
    else:
        print(f"Failed to save: {output_path}")
        return False

    # Create data list from voxel data
    colors = np.asarray(downsampled_pcd.colors) if len(downsampled_pcd.colors) > 0 else np.full((len(voxel_coordinates), 3), 0.5)
    data_list = create_voxel_data_list(voxel_coordinates, colors)
    if len(data_list) == 0:
        print(f"Warning: No voxel data generated !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print(f"Created data list with {len(data_list)} color groups {output_path}")

    # Convert data_list to sequence using precomputed_dataset logic
    if bbox_min is not None and bbox_max is not None:
        # Calculate volume dimensions from bbox
        volume_size = bbox_max - bbox_min
        voxel_dims = np.ceil(volume_size / voxel_size).astype(int)
        volume_dims = (int(voxel_dims[0]), int(voxel_dims[1]), int(voxel_dims[2]))
        
        token_sequence, seq = convert_data_list_to_sequence(data_list, volume_dims)
        if len(token_sequence) > 0:
            print(f"Converted to token sequence with {len(token_sequence)} tokens, {volume_dims}")
            
            # Save seq as PTH file
            output_path_obj = Path(output_path)
            seq_output_path = output_path_obj.parent / f"{output_path_obj.stem}_seq.pth"
            
            # Create data structure to save
            seq_data = {
                'seq': seq,
                'token_sequence': token_sequence,
                'data_list': data_list,
                'volume_dims': volume_dims,
                'voxel_size': voxel_size,
                'bbox_min': bbox_min,
                'bbox_max': bbox_max,
                'scene_name': output_path_obj.stem
            }
            
            torch.save(seq_data, str(seq_output_path))
            print(f"Saved sequence data to: {seq_output_path}")
        else:
            print("Warning: Failed to convert data_list to token sequence")
    else:
        print("Warning: bbox_min or bbox_max not provided, skipping sequence conversion")

    return True


def transform_and_save_scene_pointcloud(scene_path: Path, rotation_matrix: np.ndarray, translation_vector: np.ndarray,
                                        bbox_min: np.ndarray, bbox_max: np.ndarray) -> bool:
    """Transform a *_scene.ply point cloud, crop to bbox, and save as *_scene_aligned.ply"""
    scene_pcd = o3d.io.read_point_cloud(str(scene_path))
    if len(scene_pcd.points) == 0:
        print(f"Warning: Empty scene point cloud {scene_path.name}, skipping")
        return False

    scene_pcd.rotate(rotation_matrix, center=(0, 0, 0))
    scene_pcd.translate(translation_vector)

    aligned_scene = filter_points_by_bbox(scene_pcd, bbox_min, bbox_max)
    print(f"Scene {scene_path.name}: {len(scene_pcd.points)} -> {len(aligned_scene.points)} points inside bbox")

    output_path = scene_path.parent / f"{scene_path.stem}_aligned.ply"
    if o3d.io.write_point_cloud(str(output_path), aligned_scene):
        print(f"Saved aligned scene point cloud: {output_path}")
        return True

    print(f"Failed to save aligned scene point cloud: {output_path}")
    return False


def save_transformation_xml(rotation_matrix, translation_vector, min_bound, max_bound, voxel_size, output_path):
    """Save transformation matrices, bounding box, and voxel resolution to XML file"""
    root = ET.Element("transformation")

    # Add rotation matrix
    rotation_elem = ET.SubElement(root, "rotation_matrix")
    rotation_elem.text = " ".join([str(x) for x in rotation_matrix.flatten()])
    
    # Add translation vector
    translation_elem = ET.SubElement(root, "translation_vector")
    translation_elem.text = " ".join([str(x) for x in translation_vector])
    
    # Add bounding box
    bbox_elem = ET.SubElement(root, "bounding_box")
    min_elem = ET.SubElement(bbox_elem, "min_bound")
    min_elem.text = " ".join([str(x) for x in min_bound])
    max_elem = ET.SubElement(bbox_elem, "max_bound")
    max_elem.text = " ".join([str(x) for x in max_bound])
    
    # Add voxel resolution
    voxel_elem = ET.SubElement(root, "voxel_resolution")
    voxel_elem.text = str(voxel_size)
    
    # Calculate volume dimensions in voxels
    volume_size = max_bound - min_bound
    voxel_dims = np.ceil(volume_size / voxel_size).astype(int)
    
    volume_elem = ET.SubElement(root, "volume_dimensions")
    volume_elem.text = " ".join([str(x) for x in voxel_dims])
    
    # Create tree and write to file
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ", level=0)
    tree.write(output_path, encoding="utf-8", xml_declaration=True)
    print(f"Transformation data saved to: {output_path}")
    print(f"Volume dimensions: {voxel_dims} voxels at {voxel_size}m resolution")


def load_transformation_xml(xml_path):
    """Load rotation, translation, bbox, and voxel size from existing XML file"""
    tree = ET.parse(str(xml_path))
    root = tree.getroot()

    def parse_float_list(text, expected_length):
        if text is None:
            raise ValueError("Missing expected numeric data in transformation.xml")
        values = [float(x) for x in text.strip().split()]
        if len(values) != expected_length:
            raise ValueError(f"Expected {expected_length} values, got {len(values)}")
        return values

    rotation_values = parse_float_list(root.findtext("rotation_matrix"), 9)
    translation_values = parse_float_list(root.findtext("translation_vector"), 3)

    bbox_elem = root.find("bounding_box")
    if bbox_elem is None:
        raise ValueError("Missing bounding_box element in transformation.xml")

    min_bound_values = parse_float_list(bbox_elem.findtext("min_bound"), 3)
    max_bound_values = parse_float_list(bbox_elem.findtext("max_bound"), 3)

    voxel_text = root.findtext("voxel_resolution")
    if voxel_text is None:
        raise ValueError("Missing voxel_resolution element in transformation.xml")
    voxel_size = float(voxel_text.strip())

    return {
        'rotation_matrix': np.array(rotation_values, dtype=float).reshape(3, 3),
        'translation_vector': np.array(translation_values, dtype=float),
        'min_bound': np.array(min_bound_values, dtype=float),
        'max_bound': np.array(max_bound_values, dtype=float),
        'voxel_size': voxel_size
    }


def save_visualization_files(pcd_data, rotation_matrix, translation_vector, min_bound, max_bound, output_dir):
    """Save visualization files for the first scene with bounding box"""
    try:
        # Create a copy of the loaded point cloud
        pcd = o3d.geometry.PointCloud(pcd_data['pcd'])
        scene_file = pcd_data['file_path']
        

        
        # Apply same transformation as other scenes
        pcd.rotate(rotation_matrix, center=(0, 0, 0))
        pcd.translate(translation_vector)
        
        
        
        # Save transformed first scene
        #scene_output_path = Path(output_dir) / "first_scene_transformed.ply"
        #o3d.io.write_point_cloud(str(scene_output_path), pcd)
        #print(f"Saved transformed first scene: {scene_output_path}")
        
        # Create bounding box as line set and save
        bbox_points = np.array([
            [min_bound[0], min_bound[1], min_bound[2]],
            [min_bound[0], min_bound[1], max_bound[2]], 
            [min_bound[0], max_bound[1], min_bound[2]],
            [min_bound[0], max_bound[1], max_bound[2]],
            [max_bound[0], min_bound[1], min_bound[2]],
            [max_bound[0], min_bound[1], max_bound[2]],
            [max_bound[0], max_bound[1], min_bound[2]],
            [max_bound[0], max_bound[1], max_bound[2]]
        ])
        
        lines = [
            [0, 1], [0, 2], [0, 4], [1, 3], [1, 5], [2, 3],
            [2, 6], [3, 7], [4, 5], [4, 6], [5, 7], [6, 7]
        ]
        
        bbox_line_set = o3d.geometry.LineSet()
        bbox_line_set.points = o3d.utility.Vector3dVector(bbox_points)
        bbox_line_set.lines = o3d.utility.Vector2iVector(lines)
        
        # Create wireframe mesh for better visualization
        bbox_wireframe = o3d.geometry.TriangleMesh()
        
        # Create wireframe by adding thin cylinders for each edge
        for line in lines:
            start_point = bbox_points[line[0]]
            end_point = bbox_points[line[1]]
            
            # Create a thin cylinder for each edge
            cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=0.01, height=np.linalg.norm(end_point - start_point))
            
            # Orient the cylinder along the edge
            direction = end_point - start_point
            direction = direction / np.linalg.norm(direction)
            
            # Rotation to align cylinder with edge direction
            z_axis = np.array([0, 0, 1])
            if np.allclose(direction, z_axis):
                rotation_matrix = np.eye(3)
            elif np.allclose(direction, -z_axis):
                rotation_matrix = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, -1]])
            else:
                axis = np.cross(z_axis, direction)
                axis = axis / np.linalg.norm(axis)
                angle = np.arccos(np.dot(z_axis, direction))
                rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle(axis * angle)
            
            cylinder.rotate(rotation_matrix, center=(0, 0, 0))
            cylinder.translate(start_point + direction * np.linalg.norm(end_point - start_point) / 2)
            
            # Paint the cylinder red
            cylinder.paint_uniform_color([1, 0, 0])
            
            bbox_wireframe += cylinder
        
        # Save bounding box wireframe as PLY
        bbox_output_path = Path(output_dir) / "global_bbox.ply"
        o3d.io.write_triangle_mesh(str(bbox_output_path), bbox_wireframe)
        print(f"Saved global bounding box wireframe: {bbox_output_path}")
        
        # Create coordinate frame and save
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
        coord_output_path = Path(output_dir) / "coordinate_frame.ply"
        o3d.io.write_triangle_mesh(str(coord_output_path), coord_frame)
        print(f"Saved coordinate frame: {coord_output_path}")
        
        # Generate summary statistics
        points = np.asarray(pcd.points)
        summary_path = Path(output_dir) / "transformation_summary.txt"
        with open(summary_path, 'w') as f:
            f.write("Coordinate Transformation Summary\n")
            f.write("================================\n\n")
            f.write(f"First scene file: {scene_file.name}\n")
            f.write(f"Number of points in first scene: {len(points)}\n\n")
            f.write(f"Global bounding box (after transformation):\n")
            f.write(f"  Min: [{min_bound[0]:.6f}, {min_bound[1]:.6f}, {min_bound[2]:.6f}]\n")
            f.write(f"  Max: [{max_bound[0]:.6f}, {max_bound[1]:.6f}, {max_bound[2]:.6f}]\n")
            f.write(f"  Size: [{max_bound[0]-min_bound[0]:.6f}, {max_bound[1]-min_bound[1]:.6f}, {max_bound[2]-min_bound[2]:.6f}]\n\n")
            f.write(f"Point cloud statistics (first scene):\n")
            f.write(f"  Min: [{np.min(points, axis=0)}]\n")
            f.write(f"  Max: [{np.max(points, axis=0)}]\n")
            f.write(f"  Mean: [{np.mean(points, axis=0)}]\n")
            f.write(f"  Std: [{np.std(points, axis=0)}]\n\n")
            f.write("Generated files:\n")
            f.write("- first_scene_transformed.ply: Transformed first scene point cloud\n")
            f.write("- global_bbox.ply: Global bounding box wireframe\n")
            f.write("- coordinate_frame.ply: Coordinate frame at origin\n")
            f.write("- transformation.xml: Transformation matrices and bbox data\n")
        
        print(f"Saved transformation summary: {summary_path}")
        print("\nVisualization files saved. You can download and view them locally with:")
        print("- CloudCompare, MeshLab, or Open3D viewer")
        print("- Load all PLY files together to see the complete visualization")
        
    except Exception as e:
        print(f"Failed to save visualization files: {e}")

def main():
    # File paths
    ground_ply_path = "output/ground.ply"
    output_dir = "output/real_data/train_old"
    xml_output_path = Path(output_dir) / "transformation.xml"

    voxel_size = 0.0075
    rotation_matrix = None
    total_translation = None
    final_min_bound = None
    final_max_bound = None
    BBOX_MIN = None
    BBOX_MAX = None

    if xml_output_path.exists():
        print("Found existing transformation.xml. Loading stored transformation...")
        try:
            transform_data = load_transformation_xml(xml_output_path)
            rotation_matrix = transform_data['rotation_matrix']
            total_translation = transform_data['translation_vector']
            BBOX_MIN = transform_data['min_bound']
            BBOX_MAX = transform_data['max_bound']
            voxel_size = transform_data['voxel_size']
            print("Loaded rotation matrix and translation vector from transformation.xml")
        except Exception as e:
            print(f"Failed to load transformation.xml: {e}")
            print("Falling back to recomputing transformation parameters...")
            rotation_matrix = None
            total_translation = None
            BBOX_MIN = None
            BBOX_MAX = None

    if rotation_matrix is None or total_translation is None:
        print("Step 1: Reading ground PLY file and calculating normal...")
        try:
            ground_normal = -read_ply_ground(ground_ply_path)
            print(f"Ground normal: {ground_normal}")
        except Exception as e:
            print(f"Failed to read ground file: {e}")
            return

        print("\nStep 2: Computing coordinate transformation matrix...")
        target_z = np.array([0, 0, 1])
        rotation_matrix = compute_rotation_matrix(ground_normal, target_z)
        print("Rotation matrix:")
        print(rotation_matrix)

        BBOX_MIN = np.array([-0.3, -0.2, 0])
        BBOX_MAX = np.array([0.3, 0.2, 0.25])

    print("\nStep 3: Finding and loading all _objects_merged point cloud files...")
    pattern = "*_objects_merged.ply"
    merged_files = list(Path(output_dir).glob(pattern))
    print(f"Found {len(merged_files)} files:")
    for f in merged_files:
        print(f"  - {f}")

    loaded_pcds = load_all_pointclouds(merged_files)
    print(f"Successfully loaded {len(loaded_pcds)} point clouds")

    if rotation_matrix is None:
        print("No valid rotation matrix available. Aborting.")
        return

    if total_translation is None:
        print("\nStep 4: Computing global bounding box for all scenes...")
        try:
            temp_min_bound, temp_max_bound = compute_global_bbox_from_loaded(loaded_pcds, rotation_matrix)
            print("Global bounding box after rotation:")
            print(f"  Min: {temp_min_bound}")
            print(f"  Max: {temp_max_bound}")

            bbox_center_xy = (temp_min_bound[:2] + temp_max_bound[:2]) / 2
            bbox_min_z = temp_min_bound[2]
            translation_vector = np.array([-bbox_center_xy[0], -bbox_center_xy[1], -bbox_min_z])
            print(f"Translation vector: {translation_vector}")

            final_min_bound, final_max_bound = compute_global_bbox_after_transform_loaded(loaded_pcds, rotation_matrix, translation_vector)
            print("Final bounding box after full transformation:")
            print(f"  Min: {final_min_bound}")
            print(f"  Max: {final_max_bound}")

            final_bbox_center_xy = (final_min_bound[:2] + final_max_bound[:2]) / 2
            z_offset = -final_min_bound[2]
            final_adjustment = np.array([-final_bbox_center_xy[0], -final_bbox_center_xy[1], z_offset])

            final_min_bound += final_adjustment
            final_max_bound += final_adjustment

            total_translation = translation_vector + final_adjustment

            print(f"Final adjustment: {final_adjustment}")
            print(f"Total translation vector: {total_translation}")
            print("Adjusted final bounding box:")
            print(f"  Min: {final_min_bound}")
            print(f"  Max: {final_max_bound}")

            if not np.all(final_min_bound >= BBOX_MIN):
                print(f"ERROR: final_min_bound {final_min_bound} exceeds BBOX_MIN {BBOX_MIN}")
                print("Point cloud bounds are outside the allowed range!")
                return

            if not np.all(final_max_bound <= BBOX_MAX):
                print(f"ERROR: final_max_bound {final_max_bound} exceeds BBOX_MAX {BBOX_MAX}")
                print("Point cloud bounds are outside the allowed range!")
                return

            print("✓ Final bounds are within the allowed bbox range")

            save_transformation_xml(rotation_matrix, total_translation, BBOX_MIN, BBOX_MAX, voxel_size, str(xml_output_path))

        except Exception as e:
            print(f"Failed to compute bounding box: {e}")
            return
    else:
        print("\nUsing transformation loaded from XML. Skipping recomputation of global bounding box.")
        try:
            final_min_bound, final_max_bound = compute_global_bbox_after_transform_loaded(
                loaded_pcds, rotation_matrix, total_translation)
            print("Computed bounding box using stored transformation:")
            print(f"  Min: {final_min_bound}")
            print(f"  Max: {final_max_bound}")
        except Exception as e:
            print(f"Failed to evaluate bounding box with stored transformation: {e}")
            final_min_bound = BBOX_MIN
            final_max_bound = BBOX_MAX

    print("\nStep 5: Applying coordinate transform and translation to all point cloud files...")
    success_count = 0
    for pcd_data in loaded_pcds:
        input_file = pcd_data['file_path']
        output_filename = input_file.stem + "_aligned.ply"
        output_path = input_file.parent / output_filename

        print(f"\nProcessing: {input_file.name}")
        if transform_and_save_pointcloud(pcd_data, str(output_path), rotation_matrix, total_translation, voxel_size, BBOX_MIN, BBOX_MAX):
            success_count += 1

    print(f"\nComplete! Successfully processed {success_count}/{len(loaded_pcds)} files")
    print("Transformed files saved as *_aligned.ply")
    print("All point clouds are now aligned with Z-axis pointing up and centered at origin")

    scene_pattern = "*_scene.ply"
    scene_files = sorted(Path(output_dir).glob(scene_pattern))
    if total_translation is None or BBOX_MIN is None or BBOX_MAX is None:
        print("\nStep 6: Skipping scene alignment because transformation parameters are unavailable.")
    elif scene_files:
        print("\nStep 6: Applying transformation to scene point cloud files...")
        scene_success = 0
        for scene_path in scene_files:
            print(f"\nProcessing scene: {scene_path.name}")
            if transform_and_save_scene_pointcloud(scene_path, rotation_matrix, total_translation, BBOX_MIN, BBOX_MAX):
                scene_success += 1
        print(f"\nScene alignment complete: {scene_success}/{len(scene_files)} files aligned")
    else:
        print("\nStep 6: No *_scene.ply files found. Skipping scene alignment.")

    if loaded_pcds and final_min_bound is not None and final_max_bound is not None:
        print("\nStep 7: Saving visualization files for first scene with global bounding box...")
        save_visualization_files(loaded_pcds[0], rotation_matrix, total_translation,
                                final_min_bound, final_max_bound, output_dir)
    elif loaded_pcds:
        print("\nStep 7: Skipping visualization export because final bounds are unavailable.")


if __name__ == "__main__":
    main()
