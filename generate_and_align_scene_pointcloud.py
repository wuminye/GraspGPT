#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Combined pipeline to generate GraspNet scene point clouds (stage 1) and
align them to the ground plane (stage 2) without writing intermediate results.
"""

import argparse, os
import copy
import random
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import open3d as o3d
import torch
from graspnetAPI.graspnet import GraspNet
from graspnetAPI.utils.xmlhandler import xmlReader
from graspnetAPI.utils.utils import get_obj_pose_list

import  blender.process_grasp as mygrasp

try:
    from graspGPT.model.token_manager import get_token_manager, encode_sequence, decode_sequence
    from graspGPT.model.parser_and_serializer import Serializer, Seq, Scene, SB, CB,GB, GRASP, AMODAL, Parser, UNSEG
    from graspGPT.model.core import generate_seg_sequence
    from extract_sample_and_export import visualize_tokens 
except ImportError:
    get_token_manager = None
    Serializer = None
    Seq = None
    Scene = None
    SB = None
    CB = None
    print("Warning: Could not import GraspGPT modules. Sequence conversion will be skipped.")


@dataclass
class SceneData:
    scene_id: int
    ann_id: int
    objects_pcd: o3d.geometry.PointCloud
    scene_pcd: o3d.geometry.PointCloud
    fused_pcd: o3d.geometry.PointCloud
    output_dir: Path
    grasps: Any


GRASP_LABELS_CACHE: Dict[Tuple[str, str, int], Any] = {}
COLLISION_LABELS_CACHE: Dict[Tuple[str, str, int], Any] = {}
MERGED_OBJECTS_CACHE: Dict[Tuple[str, str, int, int], o3d.geometry.PointCloud] = {}


def merge_pointclouds(pcd_list: List[Tuple[o3d.geometry.PointCloud, int]]) -> o3d.geometry.PointCloud:
    if len(pcd_list) == 0:
        return o3d.geometry.PointCloud()

    all_xyz = []
    all_rgb = []
    for pcd, obj_id in pcd_list:
        if len(pcd.points) == 0:
            continue
        all_xyz.append(np.asarray(pcd.points))
        color = np.full((len(pcd.points), 3), obj_id / 255.0, dtype=np.float32)
        all_rgb.append(color)

    if len(all_xyz) == 0:
        return o3d.geometry.PointCloud()

    merged = o3d.geometry.PointCloud()
    merged.points = o3d.utility.Vector3dVector(np.vstack(all_xyz))
    merged.colors = o3d.utility.Vector3dVector(np.vstack(all_rgb))
    return merged


def sample_grasp_pointcloud(grasp_meshes: List[o3d.geometry.TriangleMesh]) -> Optional[o3d.geometry.PointCloud]:
    all_points = []
    for grasp_mesh in grasp_meshes:
        if hasattr(grasp_mesh, "vertices") and len(grasp_mesh.vertices) > 0:
            grasp_pcd = grasp_mesh.sample_points_uniformly(number_of_points=100)
            all_points.append(np.asarray(grasp_pcd.points))

    if not all_points:
        return None

    merged_points = np.vstack(all_points)
    grasp_colors = np.tile(np.array([[1.0, 0.0, 0.0]]), (len(merged_points), 1))

    grasp_pcd = o3d.geometry.PointCloud()
    grasp_pcd.points = o3d.utility.Vector3dVector(merged_points)
    grasp_pcd.colors = o3d.utility.Vector3dVector(grasp_colors)
    return grasp_pcd


def stage1_generate_scenes(
    args: argparse.Namespace,
    graspnet_cache: Optional[Dict[Tuple[str, str, int], GraspNet]] = None,
    grasp_labels_cache: Optional[Dict[Tuple[str, str, int], Any]] = None,
    collision_labels_cache: Optional[Dict[Tuple[str, str, int], Any]] = None,
    merged_objects_cache: Optional[
        Dict[Tuple[str, str, int, int], o3d.geometry.PointCloud]
    ] = None,
) -> List[SceneData]:
    scene_ids = args.scene_ids if args.scene_ids else [args.scene_id]
    output_dir = Path(args.outdir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: List[SceneData] = []
    grasp_labels_cache_dict = (
        grasp_labels_cache if grasp_labels_cache is not None else GRASP_LABELS_CACHE
    )
    collision_labels_cache_dict = (
        collision_labels_cache
        if collision_labels_cache is not None
        else COLLISION_LABELS_CACHE
    )
    merged_objects_cache_dict = (
        merged_objects_cache
        if merged_objects_cache is not None
        else MERGED_OBJECTS_CACHE
    )
    for scene_id in scene_ids:
        print(
            f"Stage 1: Generating point clouds for scene_{scene_id:04d}, ann_{args.ann_id:03d}"
        )

        cache_key = (args.root, args.camera, scene_id)
        graspnet: GraspNet
        if graspnet_cache is not None:
            graspnet = graspnet_cache.get(cache_key)  # type: ignore[assignment]
            if graspnet is None:
                print(
                    f"Creating GraspNet loader for scene_{scene_id:04d} (camera={args.camera})"
                )
                graspnet = GraspNet(
                    args.root,
                    camera=args.camera,
                    split="custom",
                    sceneIds=[scene_id],
                )
                graspnet_cache[cache_key] = graspnet
        else:
            graspnet = GraspNet(
                args.root,
                camera=args.camera,
                split="custom",
                sceneIds=[scene_id],
            )

        scene_dir = (
            Path(graspnet.root)
            / "scenes"
            / f"scene_{scene_id:04d}"
            / args.camera
        )
        camera_poses = np.load(scene_dir / "camera_poses.npy")
        camera_pose_ori = camera_poses[args.ann_id]
        align_mat = np.load(scene_dir / "cam0_wrt_table.npy")
        camera_pose = align_mat.dot(camera_pose_ori)
        scene_reader = xmlReader(os.path.join(scene_dir,'annotations','%04d.xml' %(args.ann_id,)))
        pose_vectors = scene_reader.getposevectorlist()

        object_ann_id = 0  # merged object clouds come from the canonical annotation
        cache_key_objects = (args.root, args.camera, scene_id)
        if cache_key_objects in merged_objects_cache_dict:
            merged_objects = o3d.geometry.PointCloud(
                merged_objects_cache_dict[cache_key_objects]
            )
        else:
            obj_pcd_list = graspnet.loadSceneModel(
                sceneId=scene_id,
                camera=args.camera,
                annId=0,
                align=args.align,
            )
            merged_objects = merge_pointclouds(obj_pcd_list)
            merged_objects_cache_dict[cache_key_objects] = merged_objects


        scene_pcd = None
        '''
        scene_pcd = graspnet.loadScenePointCloud(
            sceneId=scene_id,
            camera=args.camera,
            annId=args.ann_id,
            align=args.align,
            format="open3d",
            use_workspace=args.use_workspace,
            use_mask=not args.no_mask,
            use_inpainting=args.inpaint,
        )
        '''
        
        grasps = None
        try:
            obj_list,pose_list = get_obj_pose_list(camera_pose_ori,pose_vectors)
            cache_key_labels = (args.root, args.camera, scene_id)
            if cache_key_labels not in grasp_labels_cache_dict:
                grasp_labels_cache_dict[cache_key_labels] = graspnet.loadGraspLabels(
                    objIds=obj_list
                )
            grasp_labels = grasp_labels_cache_dict[cache_key_labels]

            if cache_key_labels not in collision_labels_cache_dict:
                collision_labels_cache_dict[cache_key_labels] = graspnet.loadCollisionLabels(
                    scene_id
                )
            collision_labels = collision_labels_cache_dict[cache_key_labels]

            grasps = graspnet.loadGrasp(
                sceneId=scene_id,
                annId=args.ann_id,
                format="6d",
                camera=args.camera,
                fric_coef_thresh=0.2,
                grasp_labels  = grasp_labels,
                collision_labels = collision_labels,
            )
            grasps = grasps.transform(camera_pose)  # 将抓取位姿转换到场景坐标系
            
            #grasps_pcd = sample_grasp_pointcloud(grasp_meshes)
        except Exception as exc:  # pylint: disable=broad-except
            print(f"Warning: Failed to sample grasps for scene_{scene_id:04d}: {exc}")


        # load fused pcd

        data = np.load('data/fusion_scenes/scene_%04d/%s/points.npy'%(scene_id,args.camera), allow_pickle=True)
        data = data.item()['xyz']
        fused_pcd = o3d.geometry.PointCloud()
        fused_pcd.points = o3d.utility.Vector3dVector(data.astype(np.float32))

        results.append(
            SceneData(
                scene_id=scene_id,
                ann_id=args.ann_id,
                objects_pcd=merged_objects,
                scene_pcd=scene_pcd,
                fused_pcd=fused_pcd,
                grasps=grasps,
                output_dir=output_dir,
            )
        )

    return results


def read_ply_ground(filepath: Path) -> np.ndarray:
    mesh = o3d.io.read_triangle_mesh(str(filepath))
    if len(mesh.vertices) == 0:
        pcd = o3d.io.read_point_cloud(str(filepath))
        if len(pcd.points) == 0:
            raise ValueError(f"Cannot read PLY file: {filepath}")

        if len(pcd.normals) > 0:
            normals = np.asarray(pcd.normals)
            ground_normal = np.mean(normals, axis=0)

            ground_normal = -ground_normal
            return ground_normal / np.linalg.norm(ground_normal)

        points = np.asarray(pcd.points)
        return fit_plane_normal(points)

    mesh.compute_vertex_normals()
    if len(mesh.vertex_normals) > 0:
        normals = np.asarray(mesh.vertex_normals)
        ground_normal = np.mean(normals, axis=0)

        ground_normal = -ground_normal
        return ground_normal / np.linalg.norm(ground_normal)

    mesh.compute_triangle_normals()
    if len(mesh.triangle_normals) > 0:
        normals = np.asarray(mesh.triangle_normals)
        ground_normal = np.mean(normals, axis=0)

        ground_normal = -ground_normal
        return ground_normal / np.linalg.norm(ground_normal)

    raise ValueError("Cannot compute ground normal")


def fit_plane_normal(points: np.ndarray) -> np.ndarray:
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    _, _, vt = np.linalg.svd(centered)
    normal = vt[-1, :]
    return normal / np.linalg.norm(normal)


def compute_rotation_matrix(from_vector: np.ndarray, to_vector: np.ndarray) -> np.ndarray:
    from_vector = from_vector / np.linalg.norm(from_vector)
    to_vector = to_vector / np.linalg.norm(to_vector)

    if np.allclose(from_vector, to_vector):
        return np.eye(3)
    if np.allclose(from_vector, -to_vector):
        if abs(from_vector[0]) < 0.9:
            perpendicular = np.array([1.0, 0.0, 0.0])
        else:
            perpendicular = np.array([0.0, 1.0, 0.0])
        perpendicular -= np.dot(perpendicular, from_vector) * from_vector
        perpendicular /= np.linalg.norm(perpendicular)
        return 2 * np.outer(perpendicular, perpendicular) - np.eye(3)

    v = np.cross(from_vector, to_vector)
    s = np.linalg.norm(v)
    c = np.dot(from_vector, to_vector)

    vx = np.array(
        [[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]],
        dtype=float,
    )

    return np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))


def prepare_loaded_pointclouds(stage1_data: List[SceneData]) -> List[Dict[str, object]]:
    loaded = []
    for record in stage1_data:
        pcd = record.objects_pcd
        point_count = len(pcd.points)
        if point_count == 0:
            print(f"Warning: scene_{record.scene_id:04d} objects point cloud is empty")
            continue

        bbox = pcd.get_axis_aligned_bounding_box()
        original_min = np.asarray(bbox.min_bound)
        original_max = np.asarray(bbox.max_bound)

        target_count = max(1, point_count // 10)
        step = max(1, point_count // target_count)
        downsampled = pcd.uniform_down_sample(step)

        print(
            f"Prepared scene_{record.scene_id:04d}_ann_{record.ann_id:03d}_objects_merged: "
            f"{point_count} -> {len(downsampled.points)} points"
        )

        prefix = f"scene_{record.scene_id:04d}_ann_{record.ann_id:03d}"
        file_path = record.output_dir / f"{prefix}_objects_merged.ply"



        grasp_group = record.grasps.nms(translation_thresh=0.02, rotation_thresh=0.4235987755982988)
        grasps = mygrasp.GraspGroup(grasp_group.grasp_group_array)
        grasps = grasps.random_sample(numGrasp=1500)
        grasp_parampc = grasps.to_parampc_dict() #{obj_id: (obj_id, parampcs of shape (N, 3, 3))}
        grasps_mesh = grasps.to_open3d_geometry_list()


        loaded.append(
            {
                "scene_id": record.scene_id,
                "ann_id": record.ann_id,
                "pcd": downsampled,
                "scene_pcd": record.scene_pcd,
                "fused_pcd": record.fused_pcd,
                "original_bbox_min": original_min,
                "original_bbox_max": original_max,
                "file_path": file_path,
                "output_prefix": prefix,
                "grasp_parampc": grasp_parampc,
                "output_dir": record.output_dir,
                "grasp_meshes": grasps_mesh,
            }
        )
    return loaded


def compute_global_bbox_from_loaded(
    loaded_pcds: List[Dict[str, object]], rotation_matrix: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    global_min = np.array([np.inf, np.inf, np.inf])
    global_max = np.array([-np.inf, -np.inf, -np.inf])

    for pcd_data in loaded_pcds:
        min_bound = pcd_data["original_bbox_min"]
        max_bound = pcd_data["original_bbox_max"]
        corners = np.array(
            [
                [min_bound[0], min_bound[1], min_bound[2]],
                [min_bound[0], min_bound[1], max_bound[2]],
                [min_bound[0], max_bound[1], min_bound[2]],
                [min_bound[0], max_bound[1], max_bound[2]],
                [max_bound[0], min_bound[1], min_bound[2]],
                [max_bound[0], min_bound[1], max_bound[2]],
                [max_bound[0], max_bound[1], min_bound[2]],
                [max_bound[0], max_bound[1], max_bound[2]],
            ]
        )
        transformed = corners @ rotation_matrix.T
        global_min = np.minimum(global_min, np.min(transformed, axis=0))
        global_max = np.maximum(global_max, np.max(transformed, axis=0))

    if np.any(np.isinf(global_min)) or np.any(np.isinf(global_max)):
        raise ValueError("No valid point clouds found")

    return global_min, global_max


def compute_global_bbox_after_transform_loaded(
    loaded_pcds: List[Dict[str, object]],
    rotation_matrix: np.ndarray,
    translation_vector: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    global_min = np.array([np.inf, np.inf, np.inf])
    global_max = np.array([-np.inf, -np.inf, -np.inf])

    for pcd_data in loaded_pcds:
        pcd = o3d.geometry.PointCloud(pcd_data["pcd"])
        pcd.rotate(rotation_matrix, center=(0, 0, 0))
        pcd.translate(translation_vector)
        bbox = pcd.get_axis_aligned_bounding_box()
        global_min = np.minimum(global_min, np.asarray(bbox.min_bound))
        global_max = np.maximum(global_max, np.asarray(bbox.max_bound))

    if np.any(np.isinf(global_min)) or np.any(np.isinf(global_max)):
        raise ValueError("No valid point clouds found")

    return global_min, global_max


def filter_points_by_bbox(
    pcd: o3d.geometry.PointCloud, bbox_min: np.ndarray, bbox_max: np.ndarray
) -> o3d.geometry.PointCloud:
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors) if len(pcd.colors) > 0 else None

    mask = (
        (points[:, 0] >= bbox_min[0])
        & (points[:, 0] <= bbox_max[0])
        & (points[:, 1] >= bbox_min[1])
        & (points[:, 1] <= bbox_max[1])
        & (points[:, 2] >= bbox_min[2])
        & (points[:, 2] <= bbox_max[2])
    )

    if np.sum(mask) / len(mask) < 0.05:
        print("Warning: 95% or more points are outside the bounding box")
        raise RuntimeError("Most points are outside the bounding box")

    filtered = o3d.geometry.PointCloud()
    filtered.points = o3d.utility.Vector3dVector(points[mask])
    if colors is not None:
        filtered.colors = o3d.utility.Vector3dVector(colors[mask])

    return filtered


def voxel_downsample_with_colors(
    pcd: o3d.geometry.PointCloud,
    voxel_size: float = 0.01,
    bbox_min: Optional[np.ndarray] = None,
    bbox_max: Optional[np.ndarray] = None,
) -> Tuple[o3d.geometry.PointCloud, List[Tuple[int, int, int]]]:
    if len(pcd.points) == 0:
        return pcd, []

    #import pdb;pdb.set_trace()
    points = np.asarray(pcd.points)
    has_colors = len(pcd.colors) > 0
    if has_colors:
        colors = np.round(np.asarray(pcd.colors) * 255.0)
    else:
        colors = np.full((len(points), 3), 127.0)

    if bbox_min is not None:
        voxel_indices = np.floor((points - bbox_min) / voxel_size).astype(int)
        grid_origin = bbox_min
    else:
        voxel_indices = np.floor(points / voxel_size).astype(int)
        grid_origin = np.zeros(3)

    voxel_dict: Dict[Tuple[int, int, int], Dict[str, List[np.ndarray]]] = {}
    for idx, voxel_idx in enumerate(voxel_indices):
        key = tuple(voxel_idx)
        if key not in voxel_dict:
            voxel_dict[key] = {"points": [], "colors": []}
        voxel_dict[key]["points"].append(points[idx])
        voxel_dict[key]["colors"].append(colors[idx])

    downsampled_points = []
    downsampled_colors = []
    voxel_coords: List[Tuple[int, int, int]] = []

    for voxel_key, voxel_data in voxel_dict.items():
        if bbox_min is not None:
            voxel_center = grid_origin + (np.array(voxel_key)) * voxel_size
        else:
            voxel_center = (np.array(voxel_key) ) * voxel_size

        if voxel_data["colors"]:
            int_colors = np.array(voxel_data["colors"]).astype(int)
            unique_colors, counts = np.unique(int_colors, axis=0, return_counts=True)
            most_frequent = unique_colors[np.argmax(counts)].astype(np.float32) / 255.0
            downsampled_colors.append(most_frequent)
            downsampled_points.append(voxel_center)
            voxel_coords.append(voxel_key)

    downsampled_pcd = o3d.geometry.PointCloud()
    downsampled_pcd.points = o3d.utility.Vector3dVector(np.array(downsampled_points))
    if downsampled_colors:
        downsampled_pcd.colors = o3d.utility.Vector3dVector(np.array(downsampled_colors))

    return downsampled_pcd, voxel_coords


def create_voxel_data_list(
    voxel_coordinates: List[Tuple[int, int, int]], colors: np.ndarray
) -> List[Tuple[int, torch.Tensor]]:
    data_list: List[Dict[str, object]] = []
    for voxel_coord, color in zip(voxel_coordinates, colors):
        color_255 = int(np.round(color[0] * 255.0))
        data_list.append({"coord": voxel_coord, "color_255": color_255})

    color_groups: Dict[int, List[Tuple[int, int, int]]] = {}
    for item in data_list:
        color_groups.setdefault(item["color_255"], []).append(item["coord"])

    result: List[Tuple[int, torch.Tensor]] = []
    for color_val in sorted(color_groups):
        coords = color_groups[color_val]
        coords.sort(key=lambda c: (c[0], c[1], c[2]))
        coord_tensor = torch.tensor(coords, dtype=torch.uint8)
        result.append((color_val, coord_tensor))

    return result


def convert_data_list_to_scene(
    data_list: List[Tuple[int, torch.Tensor]],
    volume_dims: Tuple[int, int, int] = (80, 54, 34)
) -> Optional[Scene]:
    if get_token_manager is None or Serializer is None:
        return [], None

  
    token_manager = get_token_manager()
    token_mapping = token_manager.generate_mapping(
        volume_dims[0], volume_dims[1], volume_dims[2]
    )

    sbs = []
    shuffled = data_list.copy()
    random.shuffle(shuffled)

    for color, coordinates in shuffled:
        if 0 <= color <= 87:
            shape_tag = f"object{color:02d}"
        else:
            shape_tag = "unknow"

        cbs = []
        for coord in coordinates.tolist():
            x, y, z = (int(coord[0]), int(coord[1]), int(coord[2]))
            cbs.append(CB(coord=(x, y, z)))
        cbs.sort(key=lambda cb: cb.coord)
        sbs.append(SB(tag=shape_tag, cbs=cbs))

    scene = Scene(sbs=sbs)

    return scene


def convert_PC_to_amodal_sequence(
    scene_entry: Any,
    scene: Scene,
    grasp_parampc: Any,
    rotation_matrix: np.ndarray,
    translation_vector: np.ndarray,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    voxel_size: float,
    volume_dims: Tuple[int, int, int] = (80, 54, 34)
):
    scene_pcd = scene_entry
    if len(scene_pcd.points) == 0:
        print(
            f"Warning: Empty scene point cloud for scene_{scene_entry['scene_id']:04d}_ann_{scene_entry['ann_id']:03d}, skipping"
        )
        return False

    scene_pcd.rotate(rotation_matrix, center=(0, 0, 0))
    scene_pcd.translate(translation_vector)
    aligned_scene = filter_points_by_bbox(scene_pcd, bbox_min, bbox_max)


    #o3d.io.write_point_cloud('fused_pcd.ply', aligned_scene)

    aligned_scene = np.floor((aligned_scene.points - bbox_min) / voxel_size).astype(int)

    aligned_scene = np.unique(aligned_scene, axis=0)

    cbs = []
   

    for coordinates in aligned_scene:
        x, y, z = (int(coordinates[0]), int(coordinates[1]), int(coordinates[2]))

        cbs.append(CB(coord=(x, y, z)))

    cbs.sort(key=lambda cb: cb.coord)

    new_scene = Scene(sbs=[SB(tag='incomplete', cbs=cbs)])


    seen_coords: set = set()
    combined_cbs: List[CB] = []
    for sb in scene.sbs:
        for cb in sb.cbs:
            if cb.coord in seen_coords:
                continue
            seen_coords.add(cb.coord)
            combined_cbs.append(CB(coord=cb.coord, serial=cb.serial))
    
    amodal_sb = SB(tag='unlabel', cbs=list(combined_cbs))
    new_amodal = AMODAL(sb=amodal_sb)

    local_gbs = []
    for id in grasp_parampc.keys():
      
        shape_tag = 'incomplete'  

        for parampc in grasp_parampc[id]:
            cbs = []
            for coord in parampc:
                # Ensure coordinate is a tuple of 3 integers
                if len(coord) == 3:
                    x, y, z = int(coord[0]), int(coord[1]), int(coord[2])
                    if x < 0 or y < 0 or z < 0:
                        continue
                    if x >= volume_dims[0] or y >= volume_dims[1] or z >= volume_dims[2]:
                        continue
                    
                    coord_tuple = (x, y, z)
                    cb = CB(coord=coord_tuple)
                    cbs.append(cb)
            
            # Create GB for this individual grasp if we have at least one coordinate
            if cbs:
                #cbs.sort(key=lambda cb: cb.coord)
                gb = GB(tag=shape_tag, cbs=cbs)
                local_gbs.append(gb)

    grasp = GRASP(gbs=local_gbs)

    seq = Seq(items=[new_scene, new_amodal, grasp])
    flat_tokens = Serializer.serialize(seq)
        
    return flat_tokens


def convert_PC_to_unseg_sequence(
    scene_entry: Any,
    scene: Scene,
    grasp_parampc: Any,
    rotation_matrix: np.ndarray,
    translation_vector: np.ndarray,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    voxel_size: float,
    seq_obj,
    volume_dims: Tuple[int, int, int] = (80, 54, 34)
):
    scene_pcd = scene_entry
    if len(scene_pcd.points) == 0:
        print(
            f"Warning: Empty scene point cloud for scene_{scene_entry['scene_id']:04d}_ann_{scene_entry['ann_id']:03d}, skipping"
        )
        return False

    scene_pcd.rotate(rotation_matrix, center=(0, 0, 0))
    scene_pcd.translate(translation_vector)
    aligned_scene = filter_points_by_bbox(scene_pcd, bbox_min, bbox_max)


    #o3d.io.write_point_cloud('fused_pcd.ply', aligned_scene)

    aligned_scene = np.floor((aligned_scene.points - bbox_min) / voxel_size).astype(int)

    aligned_scene = np.unique(aligned_scene, axis=0)

    cbs = []
   

    for coordinates in aligned_scene:
        x, y, z = (int(coordinates[0]), int(coordinates[1]), int(coordinates[2]))
        #if z <2:
        #    continue
        #if y <4:
        #    continue
        cbs.append(CB(coord=(x, y, z)))

    cbs.sort(key=lambda cb: cb.coord)

    new_scene = Scene(sbs=[SB(tag='unlabel', cbs=cbs)])


    ast = Parser(seq_obj).parse()

    old_scene = None
    for i in ast.items:
        if isinstance(i,Scene):
            old_scene = i
            break
    assert old_scene is not None, "no scene detected"

    new_unseg = UNSEG(sbs=old_scene.sbs)


    '''
    seen_coords: set = set()
    combined_cbs: List[CB] = []
    for sb in scene.sbs:
        for cb in sb.cbs:
            if cb.coord in seen_coords:
                continue
            seen_coords.add(cb.coord)
            combined_cbs.append(CB(coord=cb.coord, serial=cb.serial))
    
    amodal_sb = SB(tag='unlabel', cbs=list(combined_cbs))
    new_amodal = AMODAL(sb=amodal_sb)
    '''

    local_gbs = []
    for id in grasp_parampc.keys():
      
        shape_tag = f'object{id:02d}'  

        for parampc in grasp_parampc[id]:
            cbs = []
            for coord in parampc:
                # Ensure coordinate is a tuple of 3 integers
                if len(coord) == 3:
                    x, y, z = int(coord[0]), int(coord[1]), int(coord[2])
                    if x < 0 or y < 0 or z < 0:
                        continue
                    if x >= volume_dims[0] or y >= volume_dims[1] or z >= volume_dims[2]:
                        continue
                    
                    coord_tuple = (x, y, z)
                    cb = CB(coord=coord_tuple)
                    cbs.append(cb)
            
            # Create GB for this individual grasp if we have at least one coordinate
            if cbs:
                #cbs.sort(key=lambda cb: cb.coord)
                gb = GB(tag=shape_tag, cbs=cbs)
                local_gbs.append(gb)

    grasp = GRASP(gbs=local_gbs)

    seq = Seq(items=[new_scene, new_unseg, grasp])
    flat_tokens = Serializer.serialize(seq)
        
    return flat_tokens



def convert_scene_to_sequence(
    scene: Scene,
    grasp_parampc: Any = None,
    volume_dims: Tuple[int, int, int] = (80, 54, 34),
) -> Tuple[List[int], Optional[Seq]]:
    if get_token_manager is None or Serializer is None:
        return [], None

    try:
        token_manager = get_token_manager()
        token_mapping = token_manager.generate_mapping(
            volume_dims[0], volume_dims[1], volume_dims[2]
        )


        local_gbs = []
        for id in grasp_parampc.keys():
            if 0 <= id <= 87:
                shape_tag = f'object{id:02d}'  # object00 to object87
            else:
                shape_tag = 'unknow'  # fallback for out-of-range object_ids

            for parampc in grasp_parampc[id]:
                cbs = []
                for coord in parampc:
                    # Ensure coordinate is a tuple of 3 integers
                    if len(coord) == 3:
                        x, y, z = int(coord[0]), int(coord[1]), int(coord[2])
                        if x < 0 or y < 0 or z < 0:
                            continue
                        if x >= volume_dims[0] or y >= volume_dims[1] or z >= volume_dims[2]:
                            continue
                        coord_tuple = (x, y, z)
                        cb = CB(coord=coord_tuple)
                        cbs.append(cb)
                
                # Create GB for this individual grasp if we have at least one coordinate
                if cbs:
                    # Sort CBs by coordinates for consistent ordering
                    cbs.sort(key=lambda cb: cb.coord)
                    gb = GB(tag=shape_tag, cbs=cbs)
                    local_gbs.append(gb)

        grasp = GRASP(gbs=local_gbs)


        seq = Seq(items=[scene, grasp])
        flat_tokens = Serializer.serialize(seq)

        
        return flat_tokens
    except Exception as exc:  # pylint: disable=broad-except
        print(f"Warning: Could not convert data_list to sequence: {exc}")
        return  None


def transform_and_save_pointcloud(
    pcd_data: Dict[str, object],
    output_path: Path,
    rotation_matrix: np.ndarray,
    translation_vector: np.ndarray,
    voxel_size: float,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
) -> bool:
    pcd = o3d.geometry.PointCloud(pcd_data["pcd"])

    #o3d.io.write_point_cloud(str(output_path)+'_bg.ply', pcd)

    pcd.rotate(rotation_matrix, center=(0, 0, 0))
    pcd.translate(translation_vector)

    '''
    grasp_meshes = pcd_data.get("grasp_meshes")
    if grasp_meshes:
        mesh_output_dir = output_path.parent / f"{pcd_data['output_prefix']}_grasp_meshes"
        mesh_output_dir.mkdir(parents=True, exist_ok=True)
        transformed_meshes = []
        for idx, mesh in enumerate(grasp_meshes):
            mesh_copy = copy.deepcopy(mesh)
            mesh_copy.rotate(rotation_matrix, center=(0, 0, 0))
            mesh_copy.translate(translation_vector)
            mesh_path = mesh_output_dir / f"{pcd_data['output_prefix']}_grasp_{idx:03d}.ply"
            if not o3d.io.write_triangle_mesh(str(mesh_path), mesh_copy):
                print(f"Failed to save grasp mesh: {mesh_path}")
            transformed_meshes.append(mesh_copy)
        pcd_data["grasp_meshes"] = transformed_meshes
    '''


    filtered = filter_points_by_bbox(pcd, bbox_min, bbox_max)
    downsampled, voxel_coords = voxel_downsample_with_colors(
        filtered, voxel_size, bbox_min, bbox_max
    )

    print(
        f"scene_{pcd_data['scene_id']:04d}_ann_{pcd_data['ann_id']:03d}: "
        f"{len(pcd.points)} -> {len(downsampled.points)} points after voxel grid"
    )

    
    #success = o3d.io.write_point_cloud(str(output_path), downsampled)
    #if not success:
    #    print(f"Failed to save: {output_path}")
    #    return False

    colors = (
        np.asarray(downsampled.colors)
        if len(downsampled.colors) > 0
        else np.full((len(voxel_coords), 3), 0.5)
    )
    data_list = create_voxel_data_list(voxel_coords, colors)
    print(
        f"Created data list with {len(data_list)} color groups for {output_path.name}"
    )

    volume_size = bbox_max - bbox_min
    voxel_dims = np.ceil(volume_size / voxel_size).astype(int)
    volume_dims = (int(voxel_dims[0]), int(voxel_dims[1]), int(voxel_dims[2]))


    for obj_id, parampc in pcd_data["grasp_parampc"].items():
        if parampc.size == 0:
            continue

        original_shape = parampc.shape
        points = parampc.reshape(-1, 3)
        rotated = points @ rotation_matrix.T
        translated = rotated + translation_vector
        voxel_coords = np.floor((translated - bbox_min) / voxel_size).astype(int)
        pcd_data["grasp_parampc"][obj_id] = voxel_coords.reshape(original_shape)

    scene = convert_data_list_to_scene(data_list, volume_dims)

    seq_obj = convert_scene_to_sequence(scene, pcd_data['grasp_parampc'], volume_dims)

    token_manager = get_token_manager()
    token_mapping = token_manager.generate_mapping(volume_dims[0], volume_dims[1], volume_dims[2])

    
    seq_unseg = convert_PC_to_unseg_sequence(
        scene_entry=pcd_data['fused_pcd'],
        scene = scene,
        grasp_parampc=pcd_data['grasp_parampc'],
        rotation_matrix=rotation_matrix,
        translation_vector=translation_vector,
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        voxel_size=voxel_size,
        seq_obj = seq_obj,
    )

    #seq_unseg = generate_seg_sequence(seq_obj)

 
    seq_obj = encode_sequence(seq_obj, token_mapping)
    #seq_amodal = encode_sequence(seq_amodal, token_mapping) 
    seq_unseg = encode_sequence(seq_unseg, token_mapping) 
    #visualize_tokens(seq, token_mapping,volume_dims,bbox_min,voxel_size,output_dir="./output/tokens_visual")


    results = []

    '''
    results.append({
        'raw_tokens': seq_obj,
        'ind': -1,
        'repeat': -1,
        'volume_dims': volume_dims,
        'bbox_min': bbox_min,
        'bbox_max': bbox_max,
        'voxel_size': voxel_size
    })

    results.append({
        'raw_tokens': seq_amodal,
        'ind': -1,
        'repeat': -1,
        'volume_dims': volume_dims,
        'bbox_min': bbox_min,
        'bbox_max': bbox_max,
        'voxel_size': voxel_size
    })
    '''

    results.append({
        'raw_tokens': seq_unseg,
        'ind': -1,
        'repeat': -1,
        'volume_dims': volume_dims,
        'bbox_min': bbox_min,
        'bbox_max': bbox_max,
        'voxel_size': voxel_size
    })
    
    return results


def transform_and_save_scene_pointcloud(
    scene_entry: Dict[str, object],
    rotation_matrix: np.ndarray,
    translation_vector: np.ndarray,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
) -> bool:
    scene_pcd = o3d.geometry.PointCloud(scene_entry["pcd"])
    if len(scene_pcd.points) == 0:
        print(
            f"Warning: Empty scene point cloud for scene_{scene_entry['scene_id']:04d}_ann_{scene_entry['ann_id']:03d}, skipping"
        )
        return False

    scene_pcd.rotate(rotation_matrix, center=(0, 0, 0))
    scene_pcd.translate(translation_vector)
    aligned_scene = filter_points_by_bbox(scene_pcd, bbox_min, bbox_max)
    print(
        f"scene_{scene_entry['scene_id']:04d}_ann_{scene_entry['ann_id']:03d}_scene: "
        f"{len(scene_pcd.points)} -> {len(aligned_scene.points)} points inside bbox"
    )

    output_path = scene_entry["file_path"].parent / f"{scene_entry['file_path'].stem}_aligned.ply"
    if o3d.io.write_point_cloud(str(output_path), aligned_scene):
        print(f"Saved aligned scene point cloud: {output_path}")
        return True

    print(f"Failed to save aligned scene point cloud: {output_path}")
    return False


def save_transformation_xml(
    rotation_matrix: np.ndarray,
    translation_vector: np.ndarray,
    min_bound: np.ndarray,
    max_bound: np.ndarray,
    voxel_size: float,
    output_path: Path,
) -> None:
    root = ET.Element("transformation")

    rotation_elem = ET.SubElement(root, "rotation_matrix")
    rotation_elem.text = " ".join(str(x) for x in rotation_matrix.flatten())

    translation_elem = ET.SubElement(root, "translation_vector")
    translation_elem.text = " ".join(str(x) for x in translation_vector)

    bbox_elem = ET.SubElement(root, "bounding_box")
    min_elem = ET.SubElement(bbox_elem, "min_bound")
    min_elem.text = " ".join(str(x) for x in min_bound)
    max_elem = ET.SubElement(bbox_elem, "max_bound")
    max_elem.text = " ".join(str(x) for x in max_bound)

    voxel_elem = ET.SubElement(root, "voxel_resolution")
    voxel_elem.text = str(voxel_size)

    volume_size = max_bound - min_bound
    voxel_dims = np.ceil(volume_size / voxel_size).astype(int)
    volume_elem = ET.SubElement(root, "volume_dimensions")
    volume_elem.text = " ".join(str(int(x)) for x in voxel_dims)

    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ", level=0)
    tree.write(output_path, encoding="utf-8", xml_declaration=True)
    print(f"Transformation data saved to: {output_path}")
    print(f"Volume dimensions: {voxel_dims} voxels at {voxel_size}m resolution")


def load_transformation_xml(xml_path: Path) -> Dict[str, object]:
    tree = ET.parse(str(xml_path))
    root = tree.getroot()

    def parse_float_list(text: Optional[str], expected_length: int) -> List[float]:
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
        "rotation_matrix": np.array(rotation_values, dtype=float).reshape(3, 3),
        "translation_vector": np.array(translation_values, dtype=float),
        "min_bound": np.array(min_bound_values, dtype=float),
        "max_bound": np.array(max_bound_values, dtype=float),
        "voxel_size": voxel_size,
    }


def save_visualization_files(
    pcd_data: Dict[str, object],
    rotation_matrix: np.ndarray,
    translation_vector: np.ndarray,
    min_bound: np.ndarray,
    max_bound: np.ndarray,
    output_dir: Path,
) -> None:
    try:
        pcd = o3d.geometry.PointCloud(pcd_data["pcd"])
        pcd.rotate(rotation_matrix, center=(0, 0, 0))
        pcd.translate(translation_vector)

        bbox_points = np.array(
            [
                [min_bound[0], min_bound[1], min_bound[2]],
                [min_bound[0], min_bound[1], max_bound[2]],
                [min_bound[0], max_bound[1], min_bound[2]],
                [min_bound[0], max_bound[1], max_bound[2]],
                [max_bound[0], min_bound[1], min_bound[2]],
                [max_bound[0], min_bound[1], max_bound[2]],
                [max_bound[0], max_bound[1], min_bound[2]],
                [max_bound[0], max_bound[1], max_bound[2]],
            ]
        )

        lines = [
            [0, 1],
            [0, 2],
            [0, 4],
            [1, 3],
            [1, 5],
            [2, 3],
            [2, 6],
            [3, 7],
            [4, 5],
            [4, 6],
            [5, 7],
            [6, 7],
        ]

        bbox_wireframe = o3d.geometry.TriangleMesh()
        for line in lines:
            start = bbox_points[line[0]]
            end = bbox_points[line[1]]
            cylinder = o3d.geometry.TriangleMesh.create_cylinder(
                radius=0.01, height=np.linalg.norm(end - start)
            )
            direction = end - start
            direction /= np.linalg.norm(direction)
            z_axis = np.array([0.0, 0.0, 1.0])
            if np.allclose(direction, z_axis):
                rot = np.eye(3)
            elif np.allclose(direction, -z_axis):
                rot = np.diag([-1.0, -1.0, -1.0])
            else:
                axis = np.cross(z_axis, direction)
                axis /= np.linalg.norm(axis)
                angle = np.arccos(np.dot(z_axis, direction))
                rot = o3d.geometry.get_rotation_matrix_from_axis_angle(axis * angle)
            cylinder.rotate(rot, center=(0, 0, 0))
            cylinder.translate(start + direction * np.linalg.norm(end - start) / 2)
            cylinder.paint_uniform_color([1, 0, 0])
            bbox_wireframe += cylinder

        #bbox_output_path = output_dir / "global_bbox.ply"
        #o3d.io.write_triangle_mesh(str(bbox_output_path), bbox_wireframe)
        #print(f"Saved global bounding box wireframe: {bbox_output_path}")


        points = np.asarray(pcd.points)
        summary_path = output_dir / "transformation_summary.txt"
        with open(summary_path, "w", encoding="utf-8") as handle:
            handle.write("Coordinate Transformation Summary\n")
            handle.write("================================\n\n")
            handle.write(
                f"First scene id: scene_{pcd_data['scene_id']:04d}_ann_{pcd_data['ann_id']:03d}\n"
            )
            handle.write(f"Points after transform: {len(points)}\n\n")
            handle.write("Global bounding box (after transformation):\n")
            handle.write(
                f"  Min: [{min_bound[0]:.6f}, {min_bound[1]:.6f}, {min_bound[2]:.6f}]\n"
            )
            handle.write(
                f"  Max: [{max_bound[0]:.6f}, {max_bound[1]:.6f}, {max_bound[2]:.6f}]\n"
            )
            handle.write(
                f"  Size: [{(max_bound - min_bound)[0]:.6f}, {(max_bound - min_bound)[1]:.6f}, {(max_bound - min_bound)[2]:.6f}]\n\n"
            )
            handle.write("Generated files:\n")
            handle.write("- *_objects_merged_aligned.ply\n")
            handle.write("- *_objects_merged_aligned_seq.pth\n")
            handle.write("- *_scene_aligned.ply\n")
            handle.write("- global_bbox.ply\n")
            handle.write("- coordinate_frame.ply\n")
            handle.write("- transformation.xml\n")
        print(f"Saved transformation summary: {summary_path}")
    except Exception as exc:  # pylint: disable=broad-except
        print(f"Failed to save visualization files: {exc}")


def stage2_align_scenes(stage1_data: List[SceneData], args: argparse.Namespace) -> None:
    if not stage1_data:
        print("No scenes generated in stage 1. Nothing to align.")
        return

    output_dir = Path(args.outdir)
    xml_output_path = output_dir / "transformation.xml"

    voxel_size = args.voxel_size
    rotation_matrix = None
    total_translation = None
    final_min_bound = None
    final_max_bound = None
    bbox_min = None
    bbox_max = None


    if xml_output_path.exists():
        print("Found existing transformation.xml. Loading stored transformation...")
        try:
            transform_data = load_transformation_xml(xml_output_path)
            rotation_matrix = transform_data["rotation_matrix"]
            total_translation = transform_data["translation_vector"]
            bbox_min = transform_data["min_bound"]
            bbox_max = transform_data["max_bound"]
            voxel_size = transform_data["voxel_size"]
            print("Loaded rotation matrix and translation vector from transformation.xml")
        except Exception as exc:  # pylint: disable=broad-except
            print(f"Failed to load transformation.xml: {exc}")
            print("Falling back to recomputing transformation parameters...")
            rotation_matrix = None
            total_translation = None
            bbox_min = None
            bbox_max = None

    if rotation_matrix is None or total_translation is None:
        print("Stage 2: Reading ground PLY file and calculating normal...")
        ground_ply_path = Path(args.ground_ply)
        ground_normal = -read_ply_ground(ground_ply_path)
        print(f"Ground normal: {ground_normal}")

        print("Computing coordinate transformation matrix...")
        target_z = np.array([0.0, 0.0, 1.0])
        rotation_matrix = compute_rotation_matrix(ground_normal, target_z)
        print("Rotation matrix:")
        print(rotation_matrix)

        bbox_min = np.array([-0.3, -0.2, 0.0])
        bbox_max = np.array([0.3, 0.2, 0.25])

    print("Preparing Stage 1 object point clouds for alignment...")
    loaded_pcds = prepare_loaded_pointclouds(stage1_data)
    print(f"Prepared {len(loaded_pcds)} object point clouds")

    if not loaded_pcds:
        print("No valid object point clouds to align. Exiting stage 2.")
        return

    if total_translation is None:
        print("Computing global bounding box for rotated scenes...")
        temp_min_bound, temp_max_bound = compute_global_bbox_from_loaded(
            loaded_pcds, rotation_matrix
        )
        #print("Global bounding box after rotation:")
        #print(f"  Min: {temp_min_bound}")
        #print(f"  Max: {temp_max_bound}")

        bbox_center_xy = (temp_min_bound[:2] + temp_max_bound[:2]) / 2
        bbox_min_z = temp_min_bound[2]
        translation_vector = np.array(
            [-bbox_center_xy[0], -bbox_center_xy[1], -bbox_min_z]
        )
        #print(f"Translation vector: {translation_vector}")

        final_min_bound, final_max_bound = compute_global_bbox_after_transform_loaded(
            loaded_pcds, rotation_matrix, translation_vector
        )
        #print("Final bounding box after full transformation:")
        #print(f"  Min: {final_min_bound}")
        #print(f"  Max: {final_max_bound}")

        final_bbox_center_xy = (final_min_bound[:2] + final_max_bound[:2]) / 2
        z_offset = -final_min_bound[2]
        final_adjustment = np.array(
            [-final_bbox_center_xy[0], -final_bbox_center_xy[1], z_offset]
        )

        final_min_bound += final_adjustment
        final_max_bound += final_adjustment
        total_translation = translation_vector + final_adjustment

        #print(f"Final adjustment: {final_adjustment}")
        #print(f"Total translation vector: {total_translation}")
        #print("Adjusted final bounding box:")
        #print(f"  Min: {final_min_bound}")
        #print(f"  Max: {final_max_bound}")

        if not (
            np.allclose(final_min_bound, bbox_min)
            and np.allclose(final_max_bound, bbox_max)
        ):
            print(
                "Final bounds exceed allowed range; forcing them to the configured "
                "bbox and discarding points outside the workspace."
            )
        final_min_bound = bbox_min.copy()
        final_max_bound = bbox_max.copy()
        #print("Final bounds set to configured bbox:")
        #print(f"  Min: {final_min_bound}")
        #print(f"  Max: {final_max_bound}")

        save_transformation_xml(
            rotation_matrix, total_translation, bbox_min, bbox_max, voxel_size, xml_output_path
        )
    else:
        print("Using transformation loaded from XML. Skipping recomputation of bounding box.")
        try:
            final_min_bound, final_max_bound = compute_global_bbox_after_transform_loaded(
                loaded_pcds, rotation_matrix, total_translation
            )
            #print("Computed bounding box using stored transformation:")
            #print(f"  Min: {final_min_bound}")
            #print(f"  Max: {final_max_bound}")
        except Exception as exc:  # pylint: disable=broad-except
            print(
                f"Failed to evaluate bounding box with stored transformation: {exc}. "
                "Falling back to predefined limits."
            )
            final_min_bound = bbox_min
            final_max_bound = bbox_max
        else:
            final_min_bound = bbox_min.copy()
            final_max_bound = bbox_max.copy()
            #print("Final bounds set to configured bbox:")
            #print(f"  Min: {final_min_bound}")
            #print(f"  Max: {final_max_bound}")

    print("Applying coordinate transform to object point clouds...")
    success_count = 0
    results = []
    for pcd_data in loaded_pcds:
        aligned_path = pcd_data["file_path"].parent / (
            f"{pcd_data['file_path'].stem}_aligned.ply"
        )
        print(
            f"Processing scene_{pcd_data['scene_id']:04d}_ann_{pcd_data['ann_id']:03d} -> "
            f"{aligned_path.name}"
        )
        results += transform_and_save_pointcloud(
            pcd_data,
            aligned_path,
            rotation_matrix,
            total_translation,
            voxel_size,
            bbox_min,
            bbox_max,
        )
        success_count += 1

    torch.save(results, os.path.join(output_dir, f"scene_{pcd_data['scene_id']:04d}_ann_{pcd_data['ann_id']:03d}.pth"))

  
    print(
        f"Completed object alignment for {success_count}/{len(loaded_pcds)} point clouds"
    )
    '''
    scene_entries = []
    for record in stage1_data:
        prefix = f"scene_{record.scene_id:04d}_ann_{record.ann_id:03d}"
        scene_entries.append(
            {
                "scene_id": record.scene_id,
                "ann_id": record.ann_id,
                "pcd": record.scene_pcd,
                "file_path": record.output_dir / f"{prefix}_scene.ply",
                "output_prefix": prefix,
            }
        )

    if scene_entries:
        print("Applying transformation to scene point clouds...")
        scene_success = 0
        for entry in scene_entries:
            print(
                f"Processing scene_{entry['scene_id']:04d}_ann_{entry['ann_id']:03d}_scene"
            )
            if transform_and_save_scene_pointcloud(
                entry, rotation_matrix, total_translation, bbox_min, bbox_max
            ):
                scene_success += 1
        print(f"Scene alignment complete: {scene_success}/{len(scene_entries)} scenes")
    else:
        print("No scene point clouds available for alignment.")

    if loaded_pcds and final_min_bound is not None and final_max_bound is not None:
        print("Saving visualization helpers...")
        save_visualization_files(
            loaded_pcds[0],
            rotation_matrix,
            total_translation,
            final_min_bound,
            final_max_bound,
            output_dir,
        )
    elif loaded_pcds:
        print("Skipping visualization export because final bounds are unavailable.")
    '''


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default="data",
        help="GraspNet dataset root directory (contains scenes, models, etc.)",
    )
    parser.add_argument(
        "--scene_id",
        default=0,
        type=int,
        help="Scene index (scene_0000 corresponds to 0)",
    )
    parser.add_argument(
        "--scene_ids",
        nargs="+",
        type=int,
        help="Optional list of scene ids to process in a single run (requires --split all)",
    )
    parser.add_argument(
        "--split",
        choices=[
            "all",
            "train",
            "test",
            "test_seen",
            "test_similar",
            "test_novel",
        ],
        default="test_seen",
        help="Dataset split shortcut for selecting predefined scene id ranges (default: all)",
    )
    parser.add_argument(
        "--camera",
        default="kinect",
        choices=["kinect", "realsense"],
        help="Camera type",
    )
    parser.add_argument(
        "--ann_id",
        default=0,
        type=int,
        help="Annotation id (0-255) for single-run mode",
    )
    parser.add_argument(
        "--ann_ids",
        nargs="+",
        type=int,
        help="Optional list of annotation ids to iterate over",
    )
    parser.add_argument(
        "--align",
        action="store_true",
        help="Use GraspNet alignment when loading scene models and point clouds",
    )
    parser.add_argument(
        "--use_workspace",
        action="store_true",
        help="Clip reconstructed scene to workspace when loading point clouds",
    )
    parser.add_argument(
        "--no_mask",
        action="store_true",
        help="Disable mask when reconstructing scene point cloud",
    )
    parser.add_argument(
        "--inpaint",
        action="store_true",
        help="Enable depth inpainting when reconstructing scene point cloud",
    )
    parser.add_argument(
        "--outdir",
        default="output/real_data/train_m15",
        help="Directory used for all stage 2 outputs",
    )
    parser.add_argument(
        "--ground_ply",
        default="output/ground.ply",
        help="Ground plane PLY used for computing alignment normal",
    )
    parser.add_argument(
        "--voxel_size",
        default=0.0075,
        type=float,
        help="Voxel size used during stage 2 downsampling",
    )
    return parser.parse_args()


def resolve_scene_ids(args: argparse.Namespace) -> List[int]:
    split_map = {
        "all": list(range(190)),
        "train": list(range(100)),
        "test": list(range(100, 190)),
        "test_seen": list(range(100, 130)),
        "test_similar": list(range(130, 160)),
        "test_novel": list(range(160, 190)),
    }

    if args.scene_ids:
        if args.split != "all":
            raise SystemExit("--scene_ids requires --split all")
        return list(args.scene_ids)
    return split_map[args.split]


def main() -> None:
    args = parse_arguments()

    args.align = True

    scene_ids = resolve_scene_ids(args)
    #scene_ids = [100,102]
    args.scene_ids = scene_ids
    ann_ids = [x for x in range(0,255,10)]
    #ann_ids = [0]

    for scene_id in scene_ids:
        graspnet_cache: Dict[Tuple[str, str, int], GraspNet] = {}
        grasp_labels_cache: Dict[Tuple[str, str, int], Any] = {}
        collision_labels_cache: Dict[Tuple[str, str, int], Any] = {}
        merged_objects_cache: Dict[Tuple[str, str, int, int], o3d.geometry.PointCloud] = {}
        try:
            for ann_id in ann_ids:
                
                print(
                    "\n========================================\n"
                    f"Processing scene_{scene_id:04d}, ann_{ann_id:03d}"
                    "\n========================================"
                )

                combo_args = argparse.Namespace(**vars(args))
                combo_args.scene_id = scene_id
                combo_args.scene_ids = [scene_id]
                combo_args.ann_id = ann_id

                stage1_data = stage1_generate_scenes(
                    combo_args,
                    graspnet_cache,
                    grasp_labels_cache,
                    collision_labels_cache,
                    merged_objects_cache,
                )
                if not stage1_data:
                    print(
                        "Stage 1 produced no data. Skipping alignment for this combination."
                    )
                    continue

                stage2_align_scenes(stage1_data, combo_args)
        finally:
            graspnet_cache.clear()
            grasp_labels_cache.clear()
            collision_labels_cache.clear()
            merged_objects_cache.clear()


if __name__ == "__main__":
    main()