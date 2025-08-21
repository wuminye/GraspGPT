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

def voxel_downsample_with_colors(pcd, voxel_size=0.01):
    """Downsample point cloud using voxel grid with color averaging"""
    if len(pcd.points) == 0:
        return pcd
    
    # Get points and colors
    points = np.asarray(pcd.points)
    has_colors = len(pcd.colors) > 0
    assert has_colors or len(pcd.points) > 0, "Point cloud must have points or colors"
    colors = np.asarray(pcd.colors) * 255.0


    # Calculate voxel indices for each point
    voxel_indices = np.floor(points / voxel_size).astype(int)
    
    # Create unique voxel dictionary
    voxel_dict = {}
    
    for i, voxel_idx in enumerate(voxel_indices):
        voxel_key = tuple(voxel_idx)
        
        if voxel_key not in voxel_dict:
            voxel_dict[voxel_key] = {
                'points': [],
                'colors': [] if has_colors else None
            }
        
        voxel_dict[voxel_key]['points'].append(points[i])
        if has_colors:
            voxel_dict[voxel_key]['colors'].append(colors[i])
    
    # Generate downsampled points and colors
    downsampled_points = []
    downsampled_colors = []
    
    for voxel_key, voxel_data in voxel_dict.items():
        # Voxel center position
        voxel_center = (np.array(voxel_key) + 0.5) * voxel_size
        downsampled_points.append(voxel_center)
        
        # Most frequent color if colors exist
        if has_colors and voxel_data['colors']:
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
    
    if has_colors and downsampled_colors:
        downsampled_pcd.colors = o3d.utility.Vector3dVector(np.array(downsampled_colors))
    
    return downsampled_pcd

def transform_and_save_pointcloud(pcd_data, output_path, rotation_matrix, translation_vector, voxel_size=0.01):
    """Apply translation and rotation transform to loaded point cloud, downsample, and save"""
    # Create a copy to avoid modifying original
    pcd = o3d.geometry.PointCloud(pcd_data['pcd'])
    
    # Apply rotation transform
    pcd.rotate(rotation_matrix, center=(0, 0, 0))

    # Apply translation transform
    pcd.translate(translation_vector)
    
    # Apply voxel downsampling with color averaging
    downsampled_pcd = voxel_downsample_with_colors(pcd, voxel_size)
    
    print(f"Downsampled from {len(pcd.points)} to {len(downsampled_pcd.points)} points")
    
    # Save transformed and downsampled point cloud
    success = o3d.io.write_point_cloud(output_path, downsampled_pcd)
    if success:
        print(f"Saved transformed point cloud: {output_path}")
        return True
    else:
        print(f"Failed to save: {output_path}")
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
    output_dir = "output"
    
    print("Step 1: Reading ground PLY file and calculating normal...")
    try:
        ground_normal = -read_ply_ground(ground_ply_path)
        print(f"Ground normal: {ground_normal}")
    except Exception as e:
        print(f"Failed to read ground file: {e}")
        return
    
    print("\nStep 2: Computing coordinate transformation matrix...")
    # Target Z-axis direction
    target_z = np.array([0, 0, 1])
    
    # Compute rotation matrix
    rotation_matrix = compute_rotation_matrix(ground_normal, target_z)
    print(f"Rotation matrix:")
    print(rotation_matrix)
    
    print("\nStep 3: Finding and loading all _objects_merged point cloud files...")
    pattern = "*_objects_merged.ply"
    merged_files = list(Path(output_dir).glob(pattern))
    print(f"Found {len(merged_files)} files:")
    for f in merged_files:
        print(f"  - {f}")
    
    # Load all point clouds once
    loaded_pcds = load_all_pointclouds(merged_files)
    print(f"Successfully loaded {len(loaded_pcds)} point clouds")
    
    print("\nStep 4: Computing global bounding box for all scenes...")
    try:
        # First, compute bbox after rotation only to determine translation
        temp_min_bound, temp_max_bound = compute_global_bbox_from_loaded(loaded_pcds, rotation_matrix)
        print(f"Global bounding box after rotation:")
        print(f"  Min: {temp_min_bound}")
        print(f"  Max: {temp_max_bound}")
        
        # Calculate translation to center the bbox bottom plane at origin
        bbox_center_xy = (temp_min_bound[:2] + temp_max_bound[:2]) / 2
        bbox_min_z = temp_min_bound[2]
        translation_vector = np.array([-bbox_center_xy[0], -bbox_center_xy[1], -bbox_min_z])
        print(f"Translation vector: {translation_vector}")
        
        # Now compute the final bbox after both rotation and translation
        final_min_bound, final_max_bound = compute_global_bbox_after_transform_loaded(loaded_pcds, rotation_matrix, translation_vector)
        print(f"Final bounding box after full transformation:")
        print(f"  Min: {final_min_bound}")
        print(f"  Max: {final_max_bound}")
        
        # Apply final adjustment to move bottom plane to z=0 and center XY
        final_bbox_center_xy = (final_min_bound[:2] + final_max_bound[:2]) / 2
        z_offset = -final_min_bound[2]
        final_adjustment = np.array([-final_bbox_center_xy[0], -final_bbox_center_xy[1], z_offset])
        
        # Update final bounds
        final_min_bound += final_adjustment
        final_max_bound += final_adjustment
        
        # Update total translation vector
        total_translation = translation_vector + final_adjustment
        
        print(f"Final adjustment: {final_adjustment}")
        print(f"Total translation vector: {total_translation}")
        print(f"Adjusted final bounding box:")
        print(f"  Min: {final_min_bound}")
        print(f"  Max: {final_max_bound}")
        
    except Exception as e:
        print(f"Failed to compute bounding box: {e}")
        return
    
    # Save transformation data to XML (with final bbox)
    voxel_size = 0.01  # 1cm resolution
    xml_output_path = Path(output_dir) / "transformation.xml"
    save_transformation_xml(rotation_matrix, total_translation, final_min_bound, final_max_bound, voxel_size, str(xml_output_path))
    
    print("\nStep 5: Applying coordinate transform and translation to all point cloud files...")
    success_count = 0
    for pcd_data in loaded_pcds:
        # Generate output filename
        input_file = pcd_data['file_path']
        output_filename = input_file.stem + "_aligned.ply"
        output_path = input_file.parent / output_filename
        
        print(f"\nProcessing: {input_file.name}")
        if transform_and_save_pointcloud(pcd_data, str(output_path), rotation_matrix, total_translation, voxel_size):
            success_count += 1
    
    print(f"\nComplete! Successfully processed {success_count}/{len(loaded_pcds)} files")
    print("Transformed files saved as *_aligned.ply")
    print(f"All point clouds are now aligned with Z-axis pointing up and centered at origin")
    
    # Save visualization files for first scene with bounding box
    if loaded_pcds:
        print("\nStep 6: Saving visualization files for first scene with global bounding box...")
        save_visualization_files(loaded_pcds[0], rotation_matrix, total_translation, 
                                final_min_bound, final_max_bound, output_dir)

if __name__ == "__main__":
    main()