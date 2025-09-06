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
import torch
from multiprocessing import Pool, cpu_count
import trimesh
import xml.etree.ElementTree as ET

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

def load_obj_with_materials_trimesh(obj_path: str) -> List[Tuple[trimesh.Trimesh, np.ndarray]]:
    """Load OBJ with materials using trimesh"""
    scene = trimesh.load(obj_path)
    
    meshes_with_colors = []
    
    if isinstance(scene, trimesh.Scene):
        # Handle multi-mesh scene
        for name, mesh in scene.geometry.items():
            if hasattr(mesh, 'visual') and hasattr(mesh.visual, 'material'):
                if hasattr(mesh.visual.material, 'diffuse'):
                    # Get material color (RGBA -> RGB, normalize to [0,1])
                    diffuse = mesh.visual.material.diffuse
                    if len(diffuse) >= 3:
                        color = diffuse[:3] / 255.0 if diffuse.max() > 1.0 else diffuse[:3]
                    else:
                        color = np.array([0.5, 0.5, 0.5])
                else:
                    color = np.array([0.5, 0.5, 0.5])
            else:
                color = np.array([0.5, 0.5, 0.5])
            
            meshes_with_colors.append((mesh, color))
    else:
        # Single mesh
        if hasattr(scene, 'visual') and hasattr(scene.visual, 'material'):
            if hasattr(scene.visual.material, 'diffuse'):
                diffuse = scene.visual.material.diffuse
                if len(diffuse) >= 3:
                    color = diffuse[:3] / 255.0 if diffuse.max() > 1.0 else diffuse[:3]
                else:
                    color = np.array([0.5, 0.5, 0.5])
            else:
                color = np.array([0.5, 0.5, 0.5])
        else:
            color = np.array([0.5, 0.5, 0.5])
        
        meshes_with_colors.append((scene, color))
    
    return meshes_with_colors

def trimesh_to_open3d_with_colors(meshes_with_colors: List[Tuple[trimesh.Trimesh, np.ndarray]]) -> o3d.geometry.TriangleMesh:
    """Convert trimesh meshes with colors to a single Open3D mesh"""
    all_vertices = []
    all_triangles = []
    all_colors = []
    vertex_offset = 0
    
    for trimesh_mesh, color in meshes_with_colors:
        vertices = trimesh_mesh.vertices
        triangles = trimesh_mesh.faces + vertex_offset
        
        # Create vertex colors for this mesh
        vertex_colors = np.tile(color, (len(vertices), 1))
        
        all_vertices.append(vertices)
        all_triangles.append(triangles)
        all_colors.append(vertex_colors)
        
        vertex_offset += len(vertices)
    
    # Combine all meshes
    combined_vertices = np.vstack(all_vertices)
    combined_triangles = np.vstack(all_triangles)
    combined_colors = np.vstack(all_colors)
    
    # Create Open3D mesh
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(combined_vertices)
    mesh.triangles = o3d.utility.Vector3iVector(combined_triangles)
    mesh.vertex_colors = o3d.utility.Vector3dVector(combined_colors)
    
    return mesh

def mesh_to_point_cloud_with_sampling(mesh: o3d.geometry.TriangleMesh, 
                                     num_points: int = 100000) -> o3d.geometry.PointCloud:
    """Convert mesh to point cloud using uniform sampling"""
    # Sample points from mesh surface
    pcd = mesh.sample_points_uniformly(number_of_points=num_points)
    
    # If mesh has vertex colors, interpolate colors for sampled points
    #if len(mesh.vertex_colors) > 0:
    #    print(f"Sampled {len(pcd.points)} points with colors from mesh")
    #else:
    #    print(f"Sampled {len(pcd.points)} points without colors")
    
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
        #print(f"⚠️  WARNING: {len(points_outside)} points outside bbox in {scene_name}!")
        #print(f"   BBox: min={bbox_min}, max={bbox_max}")
        #for i, (idx, point) in enumerate(points_outside[:5]):  # Show first 5 outside points
        #    print(f"   Point {idx}: {point}")
        #if len(points_outside) > 5:
        #    print(f"   ... and {len(points_outside) - 5} more points outside bbox")
        return False
    else:
        #print(f"✓ All {len(points)} points are within bbox bounds")
        return True

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
    #if removed_count > 0:
    #    print(f"   Filtered out {removed_count} points outside bbox")
    
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
       
        
        # Most frequent color
        if voxel_data['colors']:
            colors = np.array(voxel_data['colors'])
            # Convert to integers and find most frequent color
            int_colors = colors.astype(int)
            unique_colors, counts = np.unique(int_colors, axis=0, return_counts=True)
            if np.max(counts) < 2:
                continue
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

def parse_transform_xml(xml_path: str) -> Dict[int, Dict]:
    """Parse XML file and extract object transform matrices"""
    transforms = {}
    
    if not os.path.exists(xml_path):
        print(f"Warning: Transform XML file not found: {xml_path}")
        return transforms
    
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        for obj_elem in root.findall('object'):
            obj_id = int(obj_elem.get('id'))
            obj_name = obj_elem.get('name')
            
            transform_data = {
                'name': obj_name,
                'transform_matrix': None,
                'position': None,
                'rotation': None,
                'scale': None
            }
            
            # Parse transformation matrix
            transform_elem = obj_elem.find('transform')
            if transform_elem is not None:
                matrix_rows = []
                for row_elem in transform_elem.findall('row'):
                    row_values = [float(x) for x in row_elem.text.split()]
                    matrix_rows.append(row_values)
                if len(matrix_rows) == 4:
                    transform_data['transform_matrix'] = np.array(matrix_rows)
            
            # Parse position
            pos_elem = obj_elem.find('position')
            if pos_elem is not None:
                transform_data['position'] = [
                    float(pos_elem.get('x')),
                    float(pos_elem.get('y')),
                    float(pos_elem.get('z'))
                ]
            
            # Parse rotation
            rot_elem = obj_elem.find('rotation')
            if rot_elem is not None:
                transform_data['rotation'] = [
                    float(rot_elem.get('x')),
                    float(rot_elem.get('y')),
                    float(rot_elem.get('z'))
                ]
            
            # Parse scale
            scale_elem = obj_elem.find('scale')
            if scale_elem is not None:
                transform_data['scale'] = [
                    float(scale_elem.get('x')),
                    float(scale_elem.get('y')),
                    float(scale_elem.get('z'))
                ]
            
            transforms[obj_id] = transform_data
        
        print(f"Loaded transforms for {len(transforms)} objects from {xml_path}")
        
    except Exception as e:
        print(f"Error parsing XML file {xml_path}: {e}")
    
    return transforms

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
        coord_tensor = torch.tensor(coords, dtype=torch.int32)
        result_list.append((color_255, coord_tensor))
    
    return result_list

def process_single_obj(args) -> Tuple[str, Optional[List[Tuple]]]:
    """Wrapper function for multiprocessing - processes a single OBJ file"""
    obj_path, output_dir, bbox_min, bbox_max, voxel_size, sample_points = args
    result = process_obj_scene(obj_path, output_dir, bbox_min, bbox_max, voxel_size, sample_points)
    return (obj_path, result)

def process_obj_scene(obj_path: str, output_dir: str, bbox_min: np.ndarray, bbox_max: np.ndarray,
                     voxel_size: float = 0.01, sample_points: int = 100000) -> Optional[List[Tuple]]:
    """Process a single OBJ scene and return voxel data as list of (color, coordinates_tensor) tuples"""
    obj_path = Path(obj_path)
    mtl_path = obj_path.with_suffix('.mtl')
    
    if not mtl_path.exists():
        print(f"Warning: MTL file not found: {mtl_path}")
        return None
    
    try:
        print(f"\nProcessing: {obj_path.name}")
        
        # Load OBJ with materials using trimesh
        meshes_with_colors = load_obj_with_materials_trimesh(str(obj_path))
        
        # Convert to single Open3D mesh
        mesh = trimesh_to_open3d_with_colors(meshes_with_colors)
        
        # Convert to point cloud
        pcd = mesh_to_point_cloud_with_sampling(mesh, sample_points)
        #print(f"Generated point cloud with {len(pcd.points)} points")
        

        
        # Check if points are within bbox before voxelization
        points = np.asarray(pcd.points)
        all_within_bbox = check_points_in_bbox(points, bbox_min, bbox_max, obj_path.name)
        
        if not all_within_bbox:
            pcd = filter_points_by_bbox(pcd, bbox_min, bbox_max)
        
        # Apply voxel downsampling with bbox
        downsampled_pcd, voxel_coordinates = voxel_downsample_with_colors(pcd, voxel_size, bbox_min, bbox_max)
        
        # Save point cloud (optional)
        #output_path = Path(output_dir) / f"{obj_path.stem}_pointcloud.ply"
        #o3d.io.write_point_cloud(str(output_path), downsampled_pcd)
        
        # Create data list from voxel data
        colors = np.asarray(downsampled_pcd.colors) if len(downsampled_pcd.colors) > 0 else np.full((len(voxel_coordinates), 3), 0.5)
        data_list = create_voxel_data_list(voxel_coordinates, colors)
        if len(data_list) == 0:
            print(f"Warning: No voxel data generated for {obj_path.name}!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(f"Created data list with {len(data_list)} color groups {obj_path.name}")
        return data_list
            
    except Exception as e:
        print(f"Error processing {obj_path.name}: {e}")
        return None

def main():
    # Configuration - use bbox from blender.py
    input_dir = "../output/synthetic_meshes"
    output_dir = "../output/pointclouds"
    voxel_size = 0.0075  # 1cm voxels
    sample_points = 10000  # Number of points to sample from each mesh
    use_multiprocessing = True  # Switch to enable/disable multiprocessing
    
    # BBox definition from blender.py (Z-axis up coordinate system)
    BBOX_MIN = np.array([-0.3, -0.2, 0])    # bbox minimum coordinates (Z-up)
    BBOX_MAX = np.array([0.3, 0.2, 0.25])     # bbox maximum coordinates (Z-up)
    
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
    
    # Prepare arguments for multiprocessing
    args_list = []
    for obj_path in sorted(obj_files):
        args_list.append((str(obj_path), output_dir, BBOX_MIN, BBOX_MAX, voxel_size, sample_points))
        
   
    
    success_count = 0
    batch_size = 5000
    batch_count = 0
    total_files = len(args_list)
    
    # Process files in batches
    for batch_start in range(0, total_files, batch_size):
        batch_end = min(batch_start + batch_size, total_files)
        current_batch_args = args_list[batch_start:batch_end]
        current_batch_data = []
        
        print(f"\nProcessing batch {batch_count}: files {batch_start+1}-{batch_end} of {total_files}")
        
        # Track successful files to maintain data-transform correspondence
        successful_files = []
        
        if use_multiprocessing:
            # Use multiprocessing to process current batch in parallel
            num_processes = min(cpu_count(), len(current_batch_args))
            print(f"Using {num_processes} processes for parallel processing")
            
            with Pool(processes=num_processes) as pool:
                results = pool.map(process_single_obj, current_batch_args)
                
                for obj_path, data_list in results:
                    if data_list is not None:
                        current_batch_data.append(data_list)
                        successful_files.append(obj_path)
                        success_count += 1
                    else:
                        print(f"Failed to process: {obj_path}")
        else:
            # Single-threaded processing for current batch
            print("Using single-threaded processing")
            
            for args in current_batch_args:
                obj_path, data_list = process_single_obj(args)
                if data_list is not None:
                    current_batch_data.append(data_list)
                    successful_files.append(obj_path)
                    success_count += 1
                else:
                    print(f"Failed to process: {obj_path}")
        
        # Save current batch data
        if current_batch_data:
            import random
            random_num = random.randint(1000, 9999)
            pth_output_path = Path(output_dir) / f"voxel_data_batch_{batch_count}_{random_num}.pth"
            
            # Collect transform data only for successfully processed files
            batch_transforms = []
            for obj_path in successful_files:
                obj_path = Path(obj_path)
                xml_path = obj_path.with_name(obj_path.stem + '_transforms.xml')
                scene_transforms = parse_transform_xml(str(xml_path))
                batch_transforms.append(scene_transforms)  # Always append (empty dict if no transforms)
                print(f"Parsed transforms for: {obj_path.stem}")

            
            data_out = {
                'voxel_size': voxel_size,
                'bbox_min': BBOX_MIN,
                'bbox_max': BBOX_MAX,
                'volume_dims': volume_dims,
                'data_lists': current_batch_data,
                'transforms': batch_transforms,  # Add transforms data
                'batch_info': {
                    'batch_number': batch_count,
                    'files_in_batch': len(current_batch_data),
                    'file_range': f"{batch_start+1}-{batch_end}"
                }
            }
            torch.save(data_out, str(pth_output_path))
            print(f"Saved batch {batch_count} with {len(current_batch_data)} data lists and {len(batch_transforms)} transform sets to: {pth_output_path}")
        
        batch_count += 1
    
    print(f"\nProcessing complete!")
    print(f"Successfully processed {success_count}/{len(obj_files)} files in {batch_count} batches")

if __name__ == "__main__":
    main()