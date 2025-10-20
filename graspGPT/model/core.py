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
    GB,
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


from collections import deque
import numpy as np
from typing import Iterable, Tuple, List, Optional

def extract_outer_shell_no_zopen(
    occupied_points: Iterable[Tuple[int, int, int]],
    grid_shape: Tuple[int, int, int] = (80, 54, 34),
    air_seed: Optional[Tuple[int, int, int]] = (0, 0, 33),
    connectivity: int = 6,
) -> np.ndarray:
    """
    使用 3D flood fill 提取体素占据体的最外层外表面点。
    注意：z轴两端封闭（不从 z=0 或 z=Z-1 方向灌入空气）。
    """
    X, Y, Z = grid_shape
    occ = np.zeros((X, Y, Z), dtype=bool)
    for x, y, z in occupied_points:
        if not (0 <= x < X and 0 <= y < Y and 0 <= z < Z):
            raise ValueError(f"Occupied point {(x,y,z)} out of bounds for grid {grid_shape}.")
        occ[x, y, z] = True

    occ_pad = np.pad(occ, pad_width=((1,1),(1,1),(1,1)), mode='constant', constant_values=False)
    visited = np.zeros_like(occ_pad, dtype=bool)

    # 邻域定义
    if connectivity == 6:
        neigh = [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]
    elif connectivity == 26:
        neigh = [(dx,dy,dz) for dx in (-1,0,1) for dy in (-1,0,1) for dz in (-1,0,1) if not (dx==dy==dz==0)]
    else:
        raise ValueError("connectivity must be 6 or 26")

    q = deque()
    PX, PY, PZ = occ_pad.shape

    def enq_if_outside_air(ax:int, ay:int, az:int):
        if 0 <= ax < PX and 0 <= ay < PY and 0 <= az < PZ and (not occ_pad[ax, ay, az]) and (not visited[ax, ay, az]):
            visited[ax, ay, az] = True
            q.append((ax, ay, az))

    # Flood fill 起点：x/y 边界（不包括 z=0, z=Z-1）
    for ax in [0, PX-1]:
        for ay in range(PY):
            for az in range(1, PZ-1):  # 跳过 z=0 和 z=Z-1
                enq_if_outside_air(ax, ay, az)
    for ay in [0, PY-1]:
        for ax in range(PX):
            for az in range(1, PZ-1):
                enq_if_outside_air(ax, ay, az)

    # 可选：已知空气种子（如 (0,0,33)）
    if air_seed is not None:
        sx, sy, sz = air_seed
        if 0 <= sx < X and 0 <= sy < Y and 0 <= sz < Z and not occ[sx, sy, sz]:
            enq_if_outside_air(sx+1, sy+1, sz+1)

    # Flood fill BFS
    while q:
        ax, ay, az = q.popleft()
        for dx, dy, dz in neigh:
            enq_if_outside_air(ax+dx, ay+dy, az+dz)

    # 去掉 padding
    outside_air = visited[1:-1, 1:-1, 1:-1]

    # 邻接判定：外部空气邻居
    def shift_and_or(target: np.ndarray, src: np.ndarray, dx:int, dy:int, dz:int):
        xs = slice(max(0, dx), X + min(0, dx))
        ys = slice(max(0, dy), Y + min(0, dy))
        zs = slice(max(0, dz), Z + min(0, dz))
        xs2 = slice(max(0, -dx), X - max(0, dx))
        ys2 = slice(max(0, -dy), Y - max(0, dy))
        zs2 = slice(max(0, -dz), Z - max(0, dz))
        target[xs, ys, zs] |= src[xs2, ys2, zs2]

    nb_outside = np.zeros_like(occ, dtype=bool)
    for dx, dy, dz in neigh:
        shift_and_or(nb_outside, outside_air, dx, dy, dz)

    shell = occ & nb_outside
    shell_coords = np.argwhere(shell)
    return shell_coords.tolist()



def generate_amodal_sequence(
    token_sequence: List[Union[str, Tuple[int, int, int]]],
    voxel_dims: Tuple[Union[int, float], Union[int, float], Union[int, float]],
    camera_resolution: Tuple[int, int] = (256, 256),
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
        2. 点云投影与深度估计：使用 point splatting 渲染，生成供反投影使用的
           深度图。
        3. 序列组织：根据深度反投影结果，重建符合语法的 token sequence。
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

        min_z_value = 40
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

    gaussian_kernel: Tuple[Tuple[int, int, int], ...] = (
        (1, 2, 1),
        (2, 4, 2),
        (1, 2, 1),
    )

    def _smooth_depth_map(values: List[List[float]]) -> List[List[float]]:
        if np is not None:
            kernel = np.asarray(gaussian_kernel, dtype=float)
            kernel_sum = kernel.sum()
            if kernel_sum > 0:
                kernel /= kernel_sum

            depth_array = np.asarray(values, dtype=float)
            finite_mask = np.isfinite(depth_array)
            depth_filled = np.where(finite_mask, depth_array, 0.0)
            mask_float = finite_mask.astype(float)


            depth_filtered = cv2.filter2D(depth_filled, -1, kernel, borderType=cv2.BORDER_REFLECT101)
            weight_map = cv2.filter2D(mask_float, -1, kernel, borderType=cv2.BORDER_REFLECT101)
        

            with np.errstate(divide='ignore', invalid='ignore'):
                smoothed_np = np.where(weight_map > 0.0, depth_filtered / weight_map, depth_array)
            return smoothed_np.tolist()


    point_infos: List[Dict[str, Any]] = []

    floor_coords = []
    for x in  range(voxel_dims[0]):
        for y in range(voxel_dims[1]):
            floor_coords.append(CB(coord=(x, y, 0)))

    for idx, cb in enumerate(combined_cbs+floor_coords):
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
        }
        point_infos.append(info)

        if cam_z <= epsilon:
            continue

        proj_x = (cam_x / cam_z) * focal_x + principal_x
        proj_y = (-cam_y / cam_z) * focal_y + principal_y
        pixel_u = int(round(proj_x))
        pixel_v = int(round(proj_y))

        info['projected'] = (proj_x, proj_y)
        info['pixel'] = (pixel_u, pixel_v)

        if not (0 <= pixel_u < width and 0 <= pixel_v < height):
            continue

        splat_radius = max(1, int(round(splat_scale / max(cam_z, epsilon))))
        splat_radius = min(splat_radius, max_splat_radius)
        info['splat_radius'] = splat_radius

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
                if existing_depth == float('inf') or cam_z < existing_depth - epsilon:
                    depth_map[v][u] = cam_z

    depth_map = _smooth_depth_map(depth_map)

    # 由深度图反投影生成 incomplete 视角的体素点云
    drop_probability = 0.05
    noise_magnitude = 0

    def _clamp(value: int, upper_bound: int) -> int:
        return max(0, min(upper_bound, value))

    max_indices = (
        max(int(round(dim_x)) - 1, 0),
        max(int(round(dim_y)) - 1, 0),
        max(int(round(dim_z)) - 1, 0),
    )

    reconstructed_voxels: set = set()
    for v in range(height):
        for u in range(width):
            depth = depth_map[v][u]
            if not math.isfinite(depth) or depth <= epsilon:
                continue

            z_cam = depth
            x_cam = ((u + 0.5) - principal_x) * z_cam / focal_x
            y_cam = -((v + 0.5) - principal_y) * z_cam / focal_y

            world_point = (
                camera_position[0] + x_cam * right_vec[0] + y_cam * up_vec[0] + z_cam * forward[0],
                camera_position[1] + x_cam * right_vec[1] + y_cam * up_vec[1] + z_cam * forward[1],
                camera_position[2] + x_cam * right_vec[2] + y_cam * up_vec[2] + z_cam * forward[2],
            )

            if (
                world_point[0] < -epsilon
                or world_point[1] < -epsilon
                or world_point[2] < -epsilon
                or world_point[0] > dim_x + epsilon
                or world_point[1] > dim_y + epsilon
                or world_point[2] > dim_z + epsilon
            ):
                continue

            discrete_coord = (
                _clamp(int(round(world_point[0])), max_indices[0]),
                _clamp(int(round(world_point[1])), max_indices[1]),
                _clamp(int(round(world_point[2])), max_indices[2]),
            )
            if discrete_coord[2] >=3:  # delete floor
                reconstructed_voxels.add(discrete_coord)

    reconstructed_voxels_list = sorted(reconstructed_voxels)

    incomplete_coords: List[Tuple[int, int, int]] = []
    for coord in reconstructed_voxels_list:
        if rng.random() < drop_probability:
            continue

        noisy_coord: List[int] = []
        for coord_value, upper_bound in zip(coord, max_indices):
            noise = rng.uniform(-noise_magnitude, noise_magnitude)
            perturbed = int(round(coord_value + noise))
            noisy_coord.append(_clamp(perturbed, upper_bound))

        incomplete_coords.append(tuple(noisy_coord))

    if not incomplete_coords:
        if reconstructed_voxels_list:
            incomplete_coords = reconstructed_voxels_list
        else:
            incomplete_coords = [cb.coord for cb in combined_cbs]

    unique_incomplete_coords: List[Tuple[int, int, int]] = []
    seen_coords: set = set()
    for coord in incomplete_coords:
        if coord in seen_coords:
            continue
        seen_coords.add(coord)
        unique_incomplete_coords.append(coord)

    incomplete_cbs = [CB(coord=coord) for coord in unique_incomplete_coords]

    

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
        'total_coords': len(combined_cbs),
        'retained_count': len(incomplete_cbs),
        'removed_count': max(0, len(reconstructed_voxels_list) - len(incomplete_cbs)),
        'splat_settings': {
            'scale': splat_scale,
            'max_radius': max_splat_radius,
        },
        'augmentation': {
            'drop_probability': drop_probability,
            'noise_magnitude': noise_magnitude,
        },
        'depth_back_projection': {
            'reconstructed': reconstructed_voxels_list,
            'retained': [cb.coord for cb in incomplete_cbs],
        },
    }
    if depth_tensor is not None:
        projection_details['depth_tensor'] = depth_tensor

    # ---- 阶段3：重新组织 token sequence -----------------------------------------------

    incomplete_cbs.sort(key=lambda cb: cb.coord)
    combined_cbs.sort(key=lambda cb: cb.coord)

    new_scene = Scene(sbs=[SB(tag='incomplete', cbs=incomplete_cbs)])
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



def crop_z_coords(tokens, max_values, del_z):
    ast = Parser(tokens).parse()
    new_items = []

    for item in ast.items:
        
        if isinstance(item,Scene):
            new_sbs = []
            for sb in item.sbs:
                new_cbs = []
                for cb in sb.cbs:
                    if cb.coord[2] < del_z:
                        continue
                    new_cbs.append(CB(coord=cb.coord, serial=None))
                new_cbs.sort(key=lambda cb: cb.coord)
                new_sbs.append(SB(tag=sb.tag, cbs=new_cbs))
            new_items.append(Scene(sbs=new_sbs))

        if isinstance(item,UNSEG):
            new_items.append(item)
            
        if isinstance(item,GRASP):
            new_items.append(item)

    ast = Seq(items=new_items)
    return Serializer.serialize(ast)

def random_translation_argument(tokens, max_values,scale=5,real_data = False):

    translation = [random.randint(-scale, scale) for _ in range(3)]
    #if real_data:
    #    translation[-1] = -1
    #else:
    translation[-1] = 0  # z 轴不变


    def _translate_sbs(sbs: List[SB]) -> List[SB]:
        new_sbs = []
        for sb in sbs:
            new_cbs = []
            for cb in sb.cbs:
                new_coord = []
                flag=False
                for c, max_v, delta in zip(cb.coord, max_values, translation):
                    new_c = c + delta
                    if new_c < 0:
                        flag=True
                        break
                    if new_c >= max_v:
                        flag=True
                        break
                    new_coord.append(new_c)
                if flag:
                    continue
                new_cbs.append(CB(coord=tuple(new_coord), serial=None))

            new_cbs.sort(key=lambda cb: cb.coord)
            if len(new_cbs) > 0 :
                new_sbs.append(SB(tag=sb.tag, cbs=new_cbs))
        if len(new_sbs) ==0:
            return sbs,False
        return new_sbs,True
 
    ast = Parser(tokens).parse()




    new_items = []

    for item in ast.items:
    
        if isinstance(item,Scene):
            new_sbs, valid = _translate_sbs(item.sbs)
            new_items.append(Scene(sbs=new_sbs))

        if isinstance(item,UNSEG):
            if valid:
                new_sbs, valid = _translate_sbs(item.sbs)
                new_items.append(UNSEG(sbs=new_sbs))
            else:
                new_items.append(item)

        if isinstance(item,GRASP):
            new_gbs = []
            for gb in item.gbs:
               
                gb_flag = False
                
                for cb in gb.cbs:
                    new_coord = []
                    flag=False
                    for c, max_v, delta in zip(cb.coord, max_values, translation):
                        new_c = c + delta
                        if new_c < 0:
                            flag=True
                            break
                        if new_c >= max_v:
                            flag=True
                            break
                        new_coord.append(new_c)
                    if flag:
                        gb_flag = True
                        break
                    cb.coord = tuple(new_coord)
                if gb_flag:
                    continue
                new_gbs.append(GB(tag=gb.tag, cbs=gb.cbs))
            
            new_items.append(GRASP(gbs=new_gbs))
    ast = Seq(items=new_items)
    return Serializer.serialize(ast)


def maybe_modify_tuple_np(t, max_values, p_modify=0.3, p_up=0.3, p_down=0.3):
    """
    使用NumPy向量化随机修改一个整数tuple，并确保结果在[0, max_values]范围内。

    参数:
        t: 原始tuple (例如 (1, 2, 3))
        max_values: 对应每个元素的最大值 (例如 (5, 10, 7))
        p_modify: 是否整体改动的概率 (默认 0.3)
        p_up: 每个元素 +1 的概率 (默认 0.3)
        p_down: 每个元素 -1 的概率 (默认 0.3)
    返回:
        修改后的tuple
    """
    t = np.array(t, dtype=int)
    max_values = np.array(max_values, dtype=int) -1  # 包括边界

    # 是否改动整个tuple
    if np.random.rand() >= p_modify:
        return tuple(t.tolist())

    # 随机扰动：每个元素独立决定 +1、-1 或不变
    r = np.random.rand(*t.shape)
    deltas = np.zeros_like(t)
    deltas[r < p_up] = 1
    deltas[(r >= p_up) & (r < p_up + p_down)] = -1

    # 应用扰动并截断到合法范围
    new_t = np.clip(t + deltas, 0, max_values).astype(np.int32).tolist()

    return tuple(new_t)

def generate_seg_sequence( token_sequence: List[Union[str, Tuple[int, int, int]]], volume_dims: Tuple[int, int, int], tags) -> List[Union[str, Tuple[int, int, int]]]:
    """将 SCENE 聚合为单个 'unlabel' 点云，并把原始数据迁移到 UNSEG。"""

    parser = Parser(token_sequence)
    original_seq = parser.parse()

    original_scene = next((item for item in original_seq.items if isinstance(item, Scene)), None)



    if original_scene is None:
        raise ValueError("输入的 token 序列缺少 SCENE 段")

    def _clone_cb(cb: CB) -> CB:
        return CB(coord=cb.coord, serial=cb.serial)

    def _clone_sb(sb: SB) -> SB:
        return SB(tag=sb.tag, cbs=[_clone_cb(cb) for cb in sb.cbs])

    unique_serials: Dict[Tuple[int, int, int], Optional[str]] = {}
    for sb in original_scene.sbs:
        for cb in sb.cbs:
            coord = cb.coord
            serial = cb.serial
            if coord not in unique_serials:
                unique_serials[coord] = serial

    if not unique_serials:
        raise ValueError("SCENE 段中没有可用的点云数据")

    scene_coords = [coord for coord in sorted(unique_serials.keys())]

    if tags.enable_flood_fill:
        scene_coords_shell = extract_outer_shell_no_zopen(scene_coords)
    else:
        scene_coords_shell = scene_coords

    if tags.add_unlabel_noise:
        unique_serials = {}
        for coord in scene_coords_shell:
            coord = maybe_modify_tuple_np(coord, max_values=volume_dims)
            if coord not in unique_serials:
                    unique_serials[coord] = None
        scene_coords_shell = [coord for coord in sorted(unique_serials.keys())]

    if tags.add_unlabel_cropping:
        merged_cbs = [CB(coord=tuple(coord), serial=None) for coord in sorted(scene_coords_shell) if random.random() >= 0.2]
    else:
        merged_cbs = [CB(coord=tuple(coord), serial=None) for coord in sorted(scene_coords_shell)]
    
    new_scene = Scene(sbs=[SB(tag='unlabel', cbs=merged_cbs)])

    if tags.sort_unseg:
        original_scene.sbs.sort(key=lambda sb: sb.cbs[0].coord if sb.cbs else (0, 0, 0))
    scene_unseg = UNSEG(sbs=[_clone_sb(sb) for sb in original_scene.sbs])





    new_items: List[Union[Scene, UNSEG, AMODAL, GRASP]] = [new_scene]

    if tags.token_mode in ["unseg_and_scene_grasp", "unseg_only"]:
        new_items.append(scene_unseg)
    elif tags.token_mode =="unseg_grasp":
        for item in original_seq.items:
            if isinstance(item, Scene):
                continue
            if isinstance(item, AMODAL):
                new_items.append(AMODAL(sb=_clone_sb(item.sb)))
            elif isinstance(item, UNSEG):
                new_items.append(UNSEG(sbs=[_clone_sb(sb) for sb in item.sbs]))
            elif isinstance(item, GRASP):
                new_gbs: List[GB] = []
                for gb in item.gbs:
                    cloned_cbs = [_clone_cb(cb) for cb in gb.cbs]
                    new_gbs.append(GB(tag='unlabel', cbs=cloned_cbs))
                new_items.append(GRASP(gbs=new_gbs))
            else:
                new_items.append(item)

    

    return Serializer.serialize(Seq(items=new_items))

def generate_amodal_seg_sequence(
    token_sequence: List[Union[str, Tuple[int, int, int]]],
    voxel_dims: Tuple[Union[int, float], Union[int, float], Union[int, float]],
    camera_resolution: Tuple[int, int] = (256, 256),
    rng: Optional[random.Random] = None,
    return_details: bool = False,
    fov_y_degrees: float = 80.0,
) -> Union[
    List[Union[str, Tuple[int, int, int]]],
    Tuple[List[Union[str, Tuple[int, int, int]]], Dict[str, Any]],
]:
    """在 AMODAL 序列基础上插入由 SCENE 转换而来的 UNSEG 段。"""

    parser = Parser(token_sequence)
    original_seq = parser.parse()
    original_scene = next((item for item in original_seq.items if isinstance(item, Scene)), None)
    if original_scene is None:
        raise ValueError("输入的 token 序列不包含 SCENE 段，无法生成 UNSEG 数据")

    def _clone_sb(sb: SB) -> SB:
        return SB(tag=sb.tag, cbs=[CB(coord=cb.coord, serial=cb.serial) for cb in sb.cbs])

    scene_unseg = UNSEG(sbs=[_clone_sb(sb) for sb in original_scene.sbs])

    amodal_result = generate_amodal_sequence(
        token_sequence,
        voxel_dims,
        camera_resolution=camera_resolution,
        rng=rng,
        return_details=return_details,
        fov_y_degrees=fov_y_degrees,
    )

    if return_details:
        new_tokens, projection_details = amodal_result
    else:
        new_tokens = amodal_result  # type: ignore[assignment]
        projection_details = None

    new_seq = Parser(new_tokens).parse()

    insert_index: Optional[int] = None
    for idx, item in enumerate(new_seq.items):
        if isinstance(item, AMODAL):
            insert_index = idx + 1
            break

    if insert_index is None:
        raise ValueError("生成的 token 序列中缺少 AMODAL 段，无法插入 UNSEG")

    new_seq.items.insert(insert_index, scene_unseg)
    modified_tokens = Serializer.serialize(new_seq)

    if return_details and projection_details is not None:
        return modified_tokens, projection_details
    return modified_tokens


def maybe_drop_amodal_or_unseg(
    seg_output: Union[
        List[Union[str, Tuple[int, int, int]]],
        Tuple[List[Union[str, Tuple[int, int, int]]], Dict[str, Any]],
    ],
    rng: Optional[random.Random] = None,
) -> Union[
    List[Union[str, Tuple[int, int, int]]],
    Tuple[List[Union[str, Tuple[int, int, int]]], Dict[str, Any]],
]:
    """随机丢弃 AMODAL 或 UNSEG 段，用于 generate_seg_sequence 的后处理。"""

    def _apply(
        tokens: List[Union[str, Tuple[int, int, int]]],
    ) -> List[Union[str, Tuple[int, int, int]]]:
        parser = Parser(tokens)
        seq = parser.parse()

        amodal_present = any(isinstance(item, AMODAL) for item in seq.items)
        unseg_present = any(isinstance(item, UNSEG) for item in seq.items)

        randomizer = rng or random
        choices: List[str] = ['noop']
        if amodal_present:
            choices.append('drop_amodal')
            choices.append('drop_amodal')
        if unseg_present:
            choices.append('drop_unseg')
            choices.append('drop_unseg')
            choices.append('drop_unseg')

        action = randomizer.choice(choices) if len(choices) > 1 else 'noop'

        filtered_items: List[Union[Scene, UNSEG, AMODAL, GRASP]] = []
        for item in seq.items:
            if action == 'drop_amodal' and isinstance(item, (AMODAL, UNSEG)):
                continue
            if action == 'drop_unseg' and isinstance(item, UNSEG):
                continue
            filtered_items.append(item)

        return Serializer.serialize(Seq(items=filtered_items))

    if isinstance(seg_output, tuple):
        tokens, details = seg_output
        return _apply(tokens), details
    return _apply(seg_output)
