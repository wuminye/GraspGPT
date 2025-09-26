import math
import random
import re
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import numpy as np  # type: ignore
except ImportError:  # pragma: no cover
    np = None  # type: ignore

try:
    import cv2  # type: ignore
except ImportError:  # pragma: no cover
    cv2 = None  # type: ignore

from .token_manager import get_token_manager
from .parser_and_serializer import (
    AMODAL,
    GRASP,
    Parser,
    SB,
    Scene,
    Serializer,
    UNSEG,
    CB,
    Seq,
)


def extract_tag_id(tag: str) -> int:
    """从tag中提取数字ID，用作颜色标识"""
    if tag.startswith('object'):
        match = re.search(r'object(\d+)', tag)
        if match:
            return int(match.group(1))
    return 0


def generate_color_from_id(tag_id: int) -> tuple:
    """根据tag ID生成RGB颜色 (0-1范围)"""
    r = (tag_id * 37) % 256 / 255.0
    g = (tag_id * 73) % 256 / 255.0  
    b = (tag_id * 109) % 256 / 255.0
    return (r, g, b)


def collect_sb_coords(ast_node: Union[Scene, SB, UNSEG, AMODAL, GRASP], coords_with_colors: List[tuple]):
    """递归收集AST中所有SB的坐标和颜色信息"""
    if isinstance(ast_node, Scene):
        for sb in ast_node.sbs:
            collect_sb_coords(sb, coords_with_colors)

    elif isinstance(ast_node, SB):
        tag_id = extract_tag_id(ast_node.tag)
        color = generate_color_from_id(tag_id)

        for cb in ast_node.cbs:
            x, y, z = cb.coord
            coords_with_colors.append((x, y, z, color[0], color[1], color[2]))

    elif isinstance(ast_node, UNSEG):
        for sb in ast_node.sbs:
            collect_sb_coords(sb, coords_with_colors)

    elif isinstance(ast_node, AMODAL):
        collect_sb_coords(ast_node.sb, coords_with_colors)

    elif isinstance(ast_node, GRASP):
        for gb in ast_node.gbs:
            collect_sb_coords(SB(tag=gb.tag, cbs=gb.cbs), coords_with_colors)


def count_sb_nodes(ast_node: Union[Scene, SB, UNSEG, AMODAL, GRASP]) -> int:
    """递归计算AST中SB节点的总数"""
    if isinstance(ast_node, Scene):
        return sum(count_sb_nodes(sb) for sb in ast_node.sbs)
    if isinstance(ast_node, SB):
        return 1
    if isinstance(ast_node, UNSEG):
        return sum(count_sb_nodes(sb) for sb in ast_node.sbs)
    if isinstance(ast_node, AMODAL):
        return count_sb_nodes(ast_node.sb)
    if isinstance(ast_node, GRASP):
        return sum(count_sb_nodes(SB(tag=gb.tag, cbs=gb.cbs)) for gb in ast_node.gbs)
    return 0


def save_voxels(sequence: list, file_path: str):
    """将sequence解析成AST，提取所有SB的坐标，保存为带颜色的点云PLY文件"""
    parser = Parser(sequence)
    ast = parser.parse()
    
    # 计算SB总数
    total_sbs = sum(count_sb_nodes(item) for item in ast.items)
    print(f"Total SBs: {total_sbs}")
    
    coords_with_colors = []
    for item in ast.items:
        collect_sb_coords(item, coords_with_colors)
    
    with open(file_path, 'w') as f:
        # PLY header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"comment Voxel point cloud generated from AST\n")
        f.write(f"comment Total SBs: {total_sbs}\n")
        f.write(f"element vertex {len(coords_with_colors)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        
        # Vertex data
        for x, y, z, r, g, b in coords_with_colors:
            red = int(r * 255)
            green = int(g * 255)
            blue = int(b * 255)
            f.write(f"{x} {y} {z} {red} {green} {blue}\n")

    return len(coords_with_colors)


def generate_amodal_sequence(
    token_sequence: List[Union[str, Tuple[int, int, int]]],
    voxel_dims: Tuple[Union[int, float], Union[int, float], Union[int, float]],
    camera_resolution: Tuple[int, int] = (192, 192),
    rng: Optional[random.Random] = None,
    return_details: bool = False,
    fov_y_degrees: float = 80.0,
) -> Union[
    List[Union[str, Tuple[int, int, int]]],
    Tuple[List[Union[str, Tuple[int, int, int]]], Dict[str, Any]],
]:
    """生成带 AMODAL 段的新 token 序列，同时输出可选的投影细节。

    函数分三个阶段执行：
        1. 目标相机参数生成：随机采样包围盒外的相机位置，构造透视相机内外参。
        2. 点云投影与遮挡处理：使用 point splatting 渲染，前景点以更大的 splat 半径
           覆盖更多像素，实现按深度的遮挡淘汰。
        3. 序列组织：根据可见 / 被遮挡结果，重建符合语法的 token sequence。
    """

    if len(voxel_dims) < 2:
        raise ValueError("voxel_dims 至少需要包含 X、Y 两个维度")

    rng = rng or random.Random()

    width, height = camera_resolution
    if width <= 0 or height <= 0:
        raise ValueError("camera_resolution 必须为正整数")

    if not (0.0 < fov_y_degrees < 180.0):
        raise ValueError("fov_y_degrees 需要位于 (0, 180) 范围内")

    fov_y = math.radians(fov_y_degrees)
    focal_y = (height / 2.0) / math.tan(fov_y / 2.0)
    focal_x = focal_y * (width / height)
    principal_x = width / 2.0
    principal_y = height / 2.0

    parser = Parser(token_sequence)
    seq = parser.parse()
    scene = next((item for item in seq.items if isinstance(item, Scene)), None)
    if scene is None:
        raise ValueError("输入的 token 序列不包含 SCENE 段")

    seen_coords: set = set()
    combined_cbs: List[CB] = []
    for sb in scene.sbs:
        for cb in sb.cbs:
            if cb.coord in seen_coords:
                continue
            seen_coords.add(cb.coord)
            combined_cbs.append(CB(coord=cb.coord, serial=cb.serial))

    if not combined_cbs:
        raise ValueError("SCENE 中没有任何坐标，无法生成 AMODAL")

    dim_x = float(voxel_dims[0])
    dim_y = float(voxel_dims[1])
    if len(voxel_dims) >= 3:
        dim_z = float(voxel_dims[2])
    else:
        dim_z = max(float(cb.coord[2]) for cb in combined_cbs)

    target_point = (dim_x / 2.0, dim_y / 2.0, 0.0)

    def _normalize(vec: Tuple[float, float, float]) -> Tuple[float, float, float]:
        norm = math.sqrt(sum(component * component for component in vec))
        if norm <= 1e-8:
            raise ValueError("方向向量长度过小，无法归一化")
        return tuple(component / norm for component in vec)

    def _cross(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> Tuple[float, float, float]:
        return (
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        )

    def _sample_camera() -> Tuple[
        Tuple[float, float, float],
        Tuple[float, float, float],
        Tuple[float, float, float],
        Tuple[float, float, float],
    ]:
        radius = max(dim_x, dim_y, dim_z)
        if radius <= 1e-6:
            raise ValueError("voxel_dims 的尺寸过小，无法确定相机半径")

        min_z_value = 0.4
        z_requirement = (min_z_value - target_point[2]) / radius
        if z_requirement >= 1.0:
            raise ValueError("无法满足相机 Z 轴阈值要求")

        cos_theta_min = max(z_requirement, 0.0)
        if cos_theta_min >= 1.0 - 1e-8:
            raise ValueError("相机位置约束过于严格：Z 轴阈值过高")

        r = rng.random()
        if r <= 1e-6:
            r = 1e-6
        cos_theta = cos_theta_min + (1.0 - cos_theta_min) * r
        cos_theta = min(cos_theta, 1.0 - 1e-9)

        sin_theta_sq = max(0.0, 1.0 - cos_theta * cos_theta)
        sin_theta = math.sqrt(sin_theta_sq)

        azimuth = rng.uniform(0.0, 2.0 * math.pi)
        offset_x = radius * sin_theta * math.cos(azimuth)
        offset_y = radius * sin_theta * math.sin(azimuth)
        offset_z = radius * cos_theta

        position = (
            target_point[0] + offset_x,
            target_point[1] + offset_y,
            target_point[2] + offset_z,
        )

        if position[2] <= min_z_value:
            raise ValueError("生成的相机位置不满足 Z 轴阈值要求")

        forward_vec = _normalize(
            (
                target_point[0] - position[0],
                target_point[1] - position[1],
                target_point[2] - position[2],
            )
        )

        world_up = (0.0, 0.0, 1.0)
        try:
            right_vec = _normalize(_cross(forward_vec, world_up))
        except ValueError:
            world_up = (0.0, 1.0, 0.0)
            right_vec = _normalize(_cross(forward_vec, world_up))

        up_vec = _normalize(_cross(right_vec, forward_vec))

        return position, forward_vec, right_vec, up_vec

    camera_position, forward, right_vec, up_vec = _sample_camera()

    camera_params: Dict[str, Any] = {
        'projection': 'perspective',
        'position': camera_position,
        'target': target_point,
        'forward': forward,
        'right': right_vec,
        'up': up_vec,
        'resolution': {'width': width, 'height': height},
        'fov_y_degrees': fov_y_degrees,
        'intrinsics': {
            'fx': focal_x,
            'fy': focal_y,
            'cx': principal_x,
            'cy': principal_y,
        },
        'bbox': {
            'min': (0.0, 0.0, 0.0),
            'max': (dim_x, dim_y, dim_z),
        },
    }

    # ---- 阶段2：point splatting 投影 -------------------------------------------------
    epsilon = 1e-6
    splat_scale = 0.75 * focal_y
    max_splat_radius = max(2, min(8, int(max(width, height) * 0.05)))

    depth_map: List[List[float]] = [[float('inf')] * width for _ in range(height)]
    owner_map: List[List[Optional[int]]] = [[None] * width for _ in range(height)]
    point_infos: List[Dict[str, Any]] = []

    for idx, cb in enumerate(combined_cbs):
        coord = cb.coord
        rel = (
            float(coord[0]) - camera_position[0],
            float(coord[1]) - camera_position[1],
            float(coord[2]) - camera_position[2],
        )

        cam_x = rel[0] * right_vec[0] + rel[1] * right_vec[1] + rel[2] * right_vec[2]
        cam_y = rel[0] * up_vec[0] + rel[1] * up_vec[1] + rel[2] * up_vec[2]
        cam_z = rel[0] * forward[0] + rel[1] * forward[1] + rel[2] * forward[2]

        info: Dict[str, Any] = {
            'index': idx,
            'cb': cb,
            'relative': rel,
            'camera_coords': (cam_x, cam_y, cam_z),
            'projected': None,
            'pixel': None,
            'splat_radius': 0,
            'visible': False,
            'reason': None,
        }
        point_infos.append(info)

        if cam_z <= epsilon:
            info['reason'] = 'behind_camera'
            continue

        proj_x = (cam_x / cam_z) * focal_x + principal_x
        proj_y = (-cam_y / cam_z) * focal_y + principal_y
        pixel_u = int(round(proj_x))
        pixel_v = int(round(proj_y))

        info['projected'] = (proj_x, proj_y)
        info['pixel'] = (pixel_u, pixel_v)

        if not (0 <= pixel_u < width and 0 <= pixel_v < height):
            info['reason'] = 'outside_fov'
            continue

        splat_radius = max(1, int(round(splat_scale / max(cam_z, epsilon))))
        splat_radius = min(splat_radius, max_splat_radius)
        info['splat_radius'] = splat_radius
        info['splat_attempted'] = True

        radius_sq = splat_radius * splat_radius
        for dv in range(-splat_radius, splat_radius + 1):
            for du in range(-splat_radius, splat_radius + 1):
                if du * du + dv * dv > radius_sq:
                    continue
                u = pixel_u + du
                v = pixel_v + dv
                if not (0 <= u < width and 0 <= v < height):
                    continue

                existing_depth = depth_map[v][u]
                existing_idx = owner_map[v][u]
                if (
                    existing_depth == float('inf')
                    or cam_z < existing_depth - epsilon
                    or (
                        abs(cam_z - existing_depth) <= epsilon
                        and (existing_idx is None or idx < existing_idx)
                    )
                ):
                    depth_map[v][u] = cam_z
                    owner_map[v][u] = idx

    coverage_counts = {idx: 0 for idx in range(len(combined_cbs))}
    for v in range(height):
        for u in range(width):
            owner_idx = owner_map[v][u]
            if owner_idx is not None:
                coverage_counts[owner_idx] += 1

    visible_set = {idx for idx, count in coverage_counts.items() if count > 0}
    for idx, info in enumerate(point_infos):
        info['covered_pixels'] = coverage_counts[idx]
        if idx in visible_set:
            info['visible'] = True
            info['reason'] = 'visible'
        else:
            if info.get('reason') is None:
                if info.get('splat_attempted'):
                    info['reason'] = 'occluded_by_depth'
                else:
                    info['reason'] = 'no_pixel_coverage'

    visible_indices = sorted(visible_set, key=lambda i: combined_cbs[i].coord)
    visible_cbs = [combined_cbs[i] for i in visible_indices]

    occluded_indices = sorted(
        set(range(len(combined_cbs))) - visible_set,
        key=lambda i: combined_cbs[i].coord,
    )
    occluded_cbs = [combined_cbs[i] for i in occluded_indices]

    if not visible_cbs:
        raise ValueError("所有坐标在投影后都被剔除，无法生成新的 SCENE")

    # 数据增强：随机删除部分可见点，并为剩余点添加坐标噪声
    drop_probability = 0.2
    noise_magnitude = 0

    def _clamp(value: int, upper_bound: int) -> int:
        return max(0, min(upper_bound, value))

    max_indices = (
        max(int(round(dim_x)) - 1, 0),
        max(int(round(dim_y)) - 1, 0),
        max(int(round(dim_z)) - 1, 0),
    )

    augmented_visible_cbs: List[CB] = []
    for cb in visible_cbs:
        if rng.random() < drop_probability:
            continue

        noisy_coord: List[int] = []
        for coord_value, upper_bound in zip(cb.coord, max_indices):
            noise = rng.uniform(-noise_magnitude, noise_magnitude)
            perturbed = int(round(coord_value + noise))
            noisy_coord.append(_clamp(perturbed, upper_bound))

        augmented_visible_cbs.append(CB(coord=tuple(noisy_coord), serial=cb.serial))

    if not augmented_visible_cbs:
        augmented_visible_cbs = [CB(coord=cb.coord, serial=cb.serial) for cb in visible_cbs]

    

    # 构造深度图（同时尝试保留 torch / numpy 表示）
    depth_tensor = None
    depth_image_numeric: Union[List[List[float]], "np.ndarray"]
    depth_metadata: Dict[str, Any]

    try:
        import torch  # type: ignore
    except ImportError:  # pragma: no cover
        torch = None  # type: ignore

    if 'torch' in locals() and torch is not None:
        depth_tensor = torch.tensor(depth_map, dtype=torch.float32).unsqueeze(0)
        depth_image_numeric = depth_tensor[0].detach().cpu().numpy()
        depth_metadata = {
            'width': width,
            'height': height,
            'fx': focal_x,
            'fy': focal_y,
            'cx': principal_x,
            'cy': principal_y,
            'format': 'torch_CHW',
            'dtype': str(depth_tensor.dtype),
        }
    else:
        if np is not None:
            depth_image_numeric = np.asarray(depth_map, dtype=float)
        else:
            depth_image_numeric = depth_map
        depth_metadata = {
            'width': width,
            'height': height,
            'fx': focal_x,
            'fy': focal_y,
            'cx': principal_x,
            'cy': principal_y,
            'format': 'list_HW',
            'dtype': 'float',
        }

    '''
    if cv2 is not None and np is not None:
        depth_for_save = np.asarray(depth_image_numeric, dtype=float)
        finite_mask = np.isfinite(depth_for_save)
        if finite_mask.any():
            d_min = depth_for_save[finite_mask].min()
            d_max = depth_for_save[finite_mask].max()
            if d_max > d_min:
                depth_norm = (depth_for_save - d_min) / (d_max - d_min)
            else:
                depth_norm = np.zeros_like(depth_for_save)
        else:
            depth_norm = np.zeros_like(depth_for_save)
        depth_vis = (np.clip(depth_norm, 0.0, 1.0) * 255).astype(np.uint8)
        cv2.imwrite('sd.jpg', depth_vis)
    #import pdb; pdb.set_trace()
    '''

    projection_details: Dict[str, Any] = {
        'camera': camera_params,
        'raw_projections': point_infos,
        'depth_image': depth_image_numeric,
        'depth_image_metadata': depth_metadata,
        'occluded': occluded_cbs,
        'visible_indices': visible_indices,
        'total_coords': len(combined_cbs),
        'retained_count': len(augmented_visible_cbs),
        'removed_count': len(combined_cbs) - len(augmented_visible_cbs),
        'splat_settings': {
            'scale': splat_scale,
            'max_radius': max_splat_radius,
        },
        'augmentation': {
            'drop_probability': drop_probability,
            'noise_magnitude': noise_magnitude,
            'visible_before': [cb.coord for cb in visible_cbs],
            'visible_after': [cb.coord for cb in augmented_visible_cbs],
        },
    }
    if depth_tensor is not None:
        projection_details['depth_tensor'] = depth_tensor

    # ---- 阶段3：重新组织 token sequence -----------------------------------------------
    
    augmented_visible_cbs.sort(key=lambda cb: cb.coord)
    combined_cbs.sort(key=lambda cb: cb.coord)
    
    new_scene = Scene(sbs=[SB(tag='incomplete', cbs=augmented_visible_cbs)])
    amodal_sb = SB(tag='unlabel', cbs=list(combined_cbs))
    new_amodal = AMODAL(sb=amodal_sb)

    new_items: List[Union[Scene, AMODAL, UNSEG, GRASP]] = [new_scene, new_amodal]
    for item in seq.items:
        if isinstance(item, UNSEG):
            new_items.append(item)
        elif isinstance(item, GRASP):
            for gb in item.gbs:
                gb.tag = 'incomplete'
            new_items.append(item)

    new_seq = Seq(items=new_items)
    serialized = Serializer.serialize(new_seq)

    if return_details:
        return serialized, projection_details
    return serialized
