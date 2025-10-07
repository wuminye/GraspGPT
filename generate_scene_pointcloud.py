#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
功能：
1) 读取 data 下的 GraspNet 数据集 scene_0000（默认相机 kinect，标注 annId=0）
2) 使用 graspnetAPI 将每个物体的3D几何按标注的位姿放回场景（loadSceneModel）
3) 重建场景点云（loadScenePointCloud）
4) 分别保存：
   - output/scene_0000_objects_merged.ply   （合并后的物体点云）
   - output/scene_0000_scene.ply            （从深度重建的整场景点云）

环境：
pip install graspnetAPI open3d
"""

import os
import argparse
import numpy as np
import open3d as o3d
from graspnetAPI.graspnet import GraspNet


def merge_pointclouds(pcd_list):
    """将多个 open3d.geometry.PointCloud 合并为一个。"""
    if len(pcd_list) == 0:
        return o3d.geometry.PointCloud()
    # 逐一拼接
    all_xyz = []
    all_rgb = []
    for p, obj_id in pcd_list:
        if len(p.points) == 0:
            continue
        all_xyz.append(np.asarray(p.points))
        # 将RGB颜色设置为obj_id的数值
        obj_id_color = np.full((len(p.points), 3), obj_id / 255.0, dtype=np.float32)
        all_rgb.append(obj_id_color)
    if len(all_xyz) == 0:
        return o3d.geometry.PointCloud()
    xyz = np.vstack(all_xyz)
    rgb = np.vstack(all_rgb)
    out = o3d.geometry.PointCloud()
    out.points = o3d.utility.Vector3dVector(xyz)
    out.colors = o3d.utility.Vector3dVector(rgb)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data", help="GraspNet 数据集根目录（包含 scenes, models 等）")
    parser.add_argument("--scene_id", default=0, type=int, help="场景编号，例如 scene_0000 -> 0")
    parser.add_argument("--camera", default="kinect", choices=["kinect", "realsense"], help="相机类型")
    parser.add_argument("--ann_id", default=0, type=int, help="标注编号（0-255）")
    parser.add_argument("--align", action="store_true", help="是否对齐到桌面坐标系（loadSceneModel/loadScenePointCloud 的 align）")
    parser.add_argument("--use_workspace", action="store_true", help="重建场景点云时是否裁剪到工作空间（loadScenePointCloud 的 use_workspace）")
    parser.add_argument("--no_mask", action="store_true", help="重建场景点云时禁用 mask（默认使用mask）")
    parser.add_argument("--inpaint", action="store_true", help="重建场景点云时对深度图做补洞（inpainting）")
    parser.add_argument("--outdir", default="output/real_data/train", help="输出目录")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # 初始化 API
    g = GraspNet(args.root, camera=args.camera, split="custom", sceneIds=[args.scene_id])  # split 对 loadSceneModel/PointCloud 影响不大

    camera_poses = np.load(os.path.join(g.root, 'scenes', 'scene_%04d' % args.scene_id, args.camera, 'camera_poses.npy'))
    camera_pose = camera_poses[args.ann_id]
    align_mat = np.load(os.path.join(g.root, 'scenes', 'scene_%04d' % args.scene_id, args.camera, 'cam0_wrt_table.npy'))
    camera_pose = align_mat.dot(camera_pose)



    scene_id = args.scene_id
    ann_id = args.ann_id

    # 1) 获取按姿态放置的“物体几何”（每个物体一个点云）
    #    注：loadSceneModel 返回的是 open3d 点云列表，已依据标注位姿变换到场景坐标系
    obj_pcd_list = g.loadSceneModel(
        sceneId=scene_id,
        camera=args.camera,
        annId=ann_id,
        align=args.align
    )

    # 合并为一个点云，便于保存/查看
    merged_objects = merge_pointclouds(obj_pcd_list)
    obj_out_path = os.path.join(args.outdir, f"scene_{scene_id:04d}_objects_merged.ply")
    o3d.io.write_point_cloud(obj_out_path, merged_objects, write_ascii=False, compressed=False, print_progress=True)
    print(f"[保存完成] 物体点云（已按姿态放置，合并）：{obj_out_path}")

    # 2) 从深度图重建场景点云
    scene_pcd = g.loadScenePointCloud(
        sceneId=scene_id,
        camera=args.camera,
        annId=ann_id,
        align=args.align,
        format="open3d",
        use_workspace=args.use_workspace,
        use_mask=not args.no_mask,
        use_inpainting=args.inpaint
    )

    scene_out_path = os.path.join(args.outdir, f"scene_{scene_id:04d}_scene.ply")
    o3d.io.write_point_cloud(scene_out_path, scene_pcd, write_ascii=False, compressed=False, print_progress=True)
    print(f"[保存完成] 场景点云：{scene_out_path}")

    _6d_grasp = g.loadGrasp(sceneId = scene_id, annId = ann_id, format = '6d', camera = 'kinect', fric_coef_thresh = 0.2)

    _6d_grasp = _6d_grasp.transform(camera_pose)  # 将抓取位姿转换到场景坐标系


    points_grasp = _6d_grasp.random_sample(numGrasp = 20).to_open3d_geometry_list()  # 采样20个抓取，转换为open3d格式点云

    # 3) 合并抓取点云和场景点云
    # 创建一个新的点云来存储合并结果
    merged_scene_grasp = o3d.geometry.PointCloud()
    
    # 添加场景点云的点和颜色
    merged_scene_grasp.points = o3d.utility.Vector3dVector(np.asarray(scene_pcd.points))
    if len(scene_pcd.colors) > 0:
        merged_scene_grasp.colors = o3d.utility.Vector3dVector(np.asarray(scene_pcd.colors))
    else:
        # 如果场景点云没有颜色，设置为白色
        scene_colors = np.ones((len(scene_pcd.points), 3))
        merged_scene_grasp.colors = o3d.utility.Vector3dVector(scene_colors)
    
    # 合并所有抓取几何体（转换为点云）并设置为红色
    all_grasp_points = []
    for grasp_mesh in points_grasp:
        # 将三角网格转换为点云（采样表面点）
        if hasattr(grasp_mesh, 'vertices') and len(grasp_mesh.vertices) > 0:
            grasp_pcd = grasp_mesh.sample_points_uniformly(number_of_points=100)
            all_grasp_points.append(np.asarray(grasp_pcd.points))
    
    if len(all_grasp_points) > 0:
        grasp_points_combined = np.vstack(all_grasp_points)
        grasp_colors = np.array([[1.0, 0.0, 0.0]] * len(grasp_points_combined))  # 红色
        
        # 4) 单独保存抓取几何体点云
        grasp_only_pcd = o3d.geometry.PointCloud()
        grasp_only_pcd.points = o3d.utility.Vector3dVector(grasp_points_combined)
        grasp_only_pcd.colors = o3d.utility.Vector3dVector(grasp_colors)
        
        grasp_out_path = os.path.join(args.outdir, f"scene_{scene_id:04d}_grasps_only.ply")
        o3d.io.write_point_cloud(grasp_out_path, grasp_only_pcd, write_ascii=False, compressed=False, print_progress=True)
        print(f"[保存完成] 抓取几何体点云：{grasp_out_path}")
        





    print("\nDone.")


if __name__ == "__main__":
    main()
