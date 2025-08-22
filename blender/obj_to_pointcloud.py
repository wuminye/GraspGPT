#!/usr/bin/env python3
"""
OBJ to Point Cloud Converter: Convert OBJ files with MTL materials to colored point clouds
using voxelization similar to align_coords.py
"""

import numpy as np
import open3d as o3d
import os
from pathlib import Path
import re
from typing import Dict, List, Tuple, Optional

def parse_mtl_file(mtl_path: str) -> Dict[str, np.ndarray]:
    """Parse MTL file and extract material colors (Kd - diffuse color)"""
    materials = {}
    current_material = None
    
    with open(mtl_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('newmtl '):
                current_material = line.split(' ', 1)[1]
            elif line.startswith('Kd ') and current_material:
                # Parse diffuse color (RGB values)
                parts = line.split()
                if len(parts) >= 4:
                    r, g, b = float(parts[1]), float(parts[2]), float(parts[3])
                    materials[current_material] = np.array([r, g, b])
    
    return materials

def load_obj_with_materials(obj_path: str, mtl_path: str) -> o3d.geometry.TriangleMesh:
    """Load OBJ file with material information and create colored mesh"""
    # Parse materials
    materials = parse_mtl_file(mtl_path)
    print(f"Found {len(materials)} materials: {list(materials.keys())}")
    
    # Load mesh using Open3D
    mesh = o3d.io.read_triangle_mesh(obj_path)
    
    if len(mesh.vertices) == 0:
        raise ValueError(f"Failed to load mesh from {obj_path}")
    
    print(f"Loaded mesh with {len(mesh.vertices)} vertices and {len(mesh.triangles)} triangles")
    
    # Parse OBJ file to get material assignments for faces
    face_materials = []
    current_material = None
    
    with open(obj_path, 'r') as f:
        face_count = 0
        for line in f:
            line = line.strip()
            if line.startswith('usemtl '):
                current_material = line.split(' ', 1)[1]
            elif line.startswith('f '):
                # Found a face, assign current material
                face_materials.append(current_material)
                face_count += 1
    
    print(f"Found {face_count} faces with material assignments")
    
    # Assign colors to vertices based on face materials
    if len(face_materials) > 0 and len(materials) > 0:
        vertex_colors = np.zeros((len(mesh.vertices), 3))
        triangle_array = np.asarray(mesh.triangles)
        
        # For each triangle, assign material color to its vertices
        for face_idx, material_name in enumerate(face_materials):
            if face_idx < len(triangle_array) and material_name in materials:
                color = materials[material_name]
                # Get vertex indices for this triangle
                vertex_indices = triangle_array[face_idx]
                # Assign color to all vertices of this triangle
                for v_idx in vertex_indices:
                    vertex_colors[v_idx] = color


        
        # Set colors to mesh
        mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
        print(f"Applied colors to mesh vertices")
    else:
        print("Warning: No materials or face assignments found, using default gray color")
        # Default gray color
        default_color = np.array([0.5, 0.5, 0.5])
        vertex_colors = np.tile(default_color, (len(mesh.vertices), 1))
        mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
    
    return mesh

def mesh_to_point_cloud_with_sampling(mesh: o3d.geometry.TriangleMesh, 
                                     num_points: int = 100000) -> o3d.geometry.PointCloud:
    """Convert mesh to point cloud using uniform sampling"""
    # Sample points from mesh surface
    pcd = mesh.sample_points_uniformly(number_of_points=num_points)
    
    # If mesh has vertex colors, interpolate colors for sampled points
    if len(mesh.vertex_colors) > 0:
        print(f"Sampled {len(pcd.points)} points with colors from mesh")
    else:
        print(f"Sampled {len(pcd.points)} points without colors")
    
    return pcd

def check_points_in_bbox(points: np.ndarray, bbox_min: np.ndarray, bbox_max: np.ndarray, scene_name: str = "") -> bool:
    """Check if all points are within the bbox and report warnings"""
    points_outside = []
    
    for i, point in enumerate(points):
        if (point[0] < bbox_min[0] or point[0] > bbox_max[0] or
            point[1] < bbox_min[1] or point[1] > bbox_max[1] or
            point[2] < bbox_min[2] or point[2] > bbox_max[2]):
            points_outside.append((i, point))
    
    if points_outside:
        print(f"⚠️  WARNING: {len(points_outside)} points outside bbox in {scene_name}!")
        print(f"   BBox: min={bbox_min}, max={bbox_max}")
        for i, (idx, point) in enumerate(points_outside[:5]):  # Show first 5 outside points
            print(f"   Point {idx}: {point}")
        if len(points_outside) > 5:
            print(f"   ... and {len(points_outside) - 5} more points outside bbox")
        return False
    else:
        print(f"✓ All {len(points)} points are within bbox bounds")
        return True

def filter_points_by_bbox(pcd: o3d.geometry.PointCloud, bbox_min: np.ndarray, bbox_max: np.ndarray) -> o3d.geometry.PointCloud:
    """Filter point cloud to keep only points within bbox"""
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors) if len(pcd.colors) > 0 else None
    
    # Find points within bbox
    mask = ((points[:, 0] >= bbox_min[0]) & (points[:, 0] <= bbox_max[0]) &
            (points[:, 1] >= bbox_min[1]) & (points[:, 1] <= bbox_max[1]) &
            (points[:, 2] >= bbox_min[2]) & (points[:, 2] <= bbox_max[2]))
    
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
    if removed_count > 0:
        print(f"   Filtered out {removed_count} points outside bbox")
    
    return filtered_pcd

def voxel_downsample_with_colors(pcd: o3d.geometry.PointCloud, voxel_size: float = 0.01, 
                                bbox_min: np.ndarray = None, bbox_max: np.ndarray = None) -> Tuple[o3d.geometry.PointCloud, List[Tuple]]:
    """Downsample point cloud using voxel grid with color averaging"""
    if len(pcd.points) == 0:
        return pcd, []
    
    # Get points and colors
    points = np.asarray(pcd.points)
    has_colors = len(pcd.colors) > 0
    
    # Check if points are within bbox
    if bbox_min is not None and bbox_max is not None:
        check_points_in_bbox(points, bbox_min, bbox_max, "during voxelization")
    
 
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
        downsampled_points.append(voxel_center)
        
        # Store integer voxel coordinates
        voxel_coordinates.append(voxel_key)
        
        # Most frequent color
        if voxel_data['colors']:
            colors = np.array(voxel_data['colors'])
            # Convert to integers and find most frequent color
            int_colors = colors.astype(int)
            unique_colors, counts = np.unique(int_colors, axis=0, return_counts=True)
            most_frequent_color = unique_colors[np.argmax(counts)]
            most_frequent_color = most_frequent_color.astype(np.float32) / 255.0  # Normalize back to [0, 1]
            downsampled_colors.append(most_frequent_color)

    # Create new point cloud
    downsampled_pcd = o3d.geometry.PointCloud()
    downsampled_pcd.points = o3d.utility.Vector3dVector(np.array(downsampled_points))
    
    if downsampled_colors:
        downsampled_pcd.colors = o3d.utility.Vector3dVector(np.array(downsampled_colors))
    
    return downsampled_pcd, voxel_coordinates

def save_voxel_coordinates_to_txt(voxel_coordinates: List[Tuple], colors: np.ndarray, 
                                 output_path: str, bbox_min: np.ndarray, bbox_max: np.ndarray, 
                                 voxel_size: float) -> None:
    """Save voxel coordinates and colors to txt file"""
    # Calculate volume dimensions
    volume_size = bbox_max - bbox_min
    volume_dims = np.ceil(volume_size / voxel_size).astype(int)
    
    with open(output_path, 'w') as f:
        # Write header with volume information
        f.write(f"# Volume dimensions (voxels): {volume_dims[0]} {volume_dims[1]} {volume_dims[2]}\n")
        f.write(f"# Voxel size: {voxel_size}\n")
        f.write(f"# BBox min: {bbox_min[0]} {bbox_min[1]} {bbox_min[2]}\n")
        f.write(f"# BBox max: {bbox_max[0]} {bbox_max[1]} {bbox_max[2]}\n")
        f.write(f"# Format: voxel_x voxel_y voxel_z color_r color_g color_b\n")
        
        # Write voxel data
        for i, (voxel_coord, color) in enumerate(zip(voxel_coordinates, colors)):
            # Convert color from [0,1] to [0,255] range
            color_255 = np.round(color * 255.0).astype(int)
            f.write(f"{voxel_coord[0]} {voxel_coord[1]} {voxel_coord[2]} {color_255[0]}\n")

def process_obj_scene(obj_path: str, output_dir: str, bbox_min: np.ndarray, bbox_max: np.ndarray,
                     voxel_size: float = 0.01, sample_points: int = 100000) -> bool:
    """Process a single OBJ scene and save as colored point cloud"""
    obj_path = Path(obj_path)
    mtl_path = obj_path.with_suffix('.mtl')
    
    if not mtl_path.exists():
        print(f"Warning: MTL file not found: {mtl_path}")
        return False
    
    try:
        print(f"\nProcessing: {obj_path.name}")
        
        # Load OBJ with materials
        mesh = load_obj_with_materials(str(obj_path), str(mtl_path))
        
        # Convert to point cloud
        pcd = mesh_to_point_cloud_with_sampling(mesh, sample_points)
        print(f"Generated point cloud with {len(pcd.points)} points")
        
        # Save original sampled point cloud
        original_pcd_path = Path(output_dir) / f"{obj_path.stem}_sampled.ply"
        o3d.io.write_point_cloud(str(original_pcd_path), pcd)
        print(f"Saved sampled point cloud: {original_pcd_path}")
        
        # Check if points are within bbox before voxelization
        points = np.asarray(pcd.points)
        print(f"Checking if points are within bbox for {obj_path.name}...")
        all_within_bbox = check_points_in_bbox(points, bbox_min, bbox_max, obj_path.name)
        
        if not all_within_bbox:
            print(f"⚠️  Some points in {obj_path.name} are outside the expected bbox!")
            print(f"   Filtering points to keep only those within bbox...")
            pcd = filter_points_by_bbox(pcd, bbox_min, bbox_max)
            print(f"   Point cloud now has {len(pcd.points)} points within bbox")
        
        # Apply voxel downsampling with bbox
        downsampled_pcd, voxel_coordinates = voxel_downsample_with_colors(pcd, voxel_size, bbox_min, bbox_max)
        print(f"Downsampled to {len(downsampled_pcd.points)} points using {voxel_size}m voxels")
        
        # Save point cloud
        output_path = Path(output_dir) / f"{obj_path.stem}_pointcloud.ply"
        success = o3d.io.write_point_cloud(str(output_path), downsampled_pcd)
        
        # Save voxel coordinates to txt file
        txt_output_path = Path(output_dir) / f"{obj_path.stem}_voxels.txt"
        colors = np.asarray(downsampled_pcd.colors) if len(downsampled_pcd.colors) > 0 else np.full((len(voxel_coordinates), 3), 0.5)
        save_voxel_coordinates_to_txt(voxel_coordinates, colors, str(txt_output_path), bbox_min, bbox_max, voxel_size)
        
        if success:
            print(f"Saved colored point cloud: {output_path}")
            print(f"Saved voxel coordinates: {txt_output_path}")
            return True
        else:
            print(f"Failed to save: {output_path}")
            return False
            
    except Exception as e:
        print(f"Error processing {obj_path.name}: {e}")
        return False

def main():
    # Configuration - use bbox from blender.py
    input_dir = "../output/synthetic_meshes"
    output_dir = "../output/pointclouds"
    voxel_size = 0.01  # 1cm voxels
    sample_points = 10000  # Number of points to sample from each mesh
    
    # BBox definition from blender.py (Z-axis up coordinate system)
    BBOX_MIN = np.array([-0.27, -0.18, 0])    # bbox minimum coordinates (Z-up)
    BBOX_MAX = np.array([0.27, 0.18, 0.2])     # bbox maximum coordinates (Z-up)
    
    # Calculate and print volume information
    volume_size = BBOX_MAX - BBOX_MIN
    volume_dims = np.ceil(volume_size / voxel_size).astype(int)
    print(f"Volume dimensions: {volume_dims} voxels ({volume_size} meters)")
    print(f"Total voxels in volume: {np.prod(volume_dims)}")
    print(f"Voxel size: {voxel_size}m")
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    # Find all OBJ files
    obj_files = list(Path(input_dir).glob("*.obj"))
    print(f"Found {len(obj_files)} OBJ files to process")
    
    # Process each OBJ file
    success_count = 0
    for obj_path in sorted(obj_files):
        if process_obj_scene(str(obj_path), output_dir, BBOX_MIN, BBOX_MAX, voxel_size, sample_points):
            success_count += 1
        break
        
    
    print(f"\nProcessing complete!")
    print(f"Successfully processed {success_count}/{len(obj_files)} files")
    print(f"Colored point clouds saved in: {output_dir}")

if __name__ == "__main__":
    main()