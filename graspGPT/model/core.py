import re
from typing import List, Union
from .token_manager import get_token_manager
from .parser_and_serializer import Parser, Serializer, SB, UNSEG, INPAINT, AMODAL


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


def collect_sb_coords(ast_node: Union[SB, UNSEG, INPAINT, AMODAL], coords_with_colors: List[tuple]):
    """递归收集AST中所有SB的坐标和颜色信息"""
    if isinstance(ast_node, SB):
        tag_id = extract_tag_id(ast_node.tag)
        color = generate_color_from_id(tag_id)
        
        for cb in ast_node.cbs:
            x, y, z = cb.coord
            coords_with_colors.append((x, y, z, color[0], color[1], color[2]))
    
    elif isinstance(ast_node, UNSEG):
        for sb in ast_node.sbs:
            collect_sb_coords(sb, coords_with_colors)
    
    elif isinstance(ast_node, INPAINT):
        if ast_node.sb is not None:
            collect_sb_coords(ast_node.sb, coords_with_colors)
    
    elif isinstance(ast_node, AMODAL):
        for sb in ast_node.fragment_sbs:
            collect_sb_coords(sb, coords_with_colors)
        for sb in ast_node.amodal_sbs:
            collect_sb_coords(sb, coords_with_colors)


def count_sb_nodes(ast_node: Union[SB, UNSEG, INPAINT, AMODAL]) -> int:
    """递归计算AST中SB节点的总数"""
    if isinstance(ast_node, SB):
        return 1
    elif isinstance(ast_node, UNSEG):
        return sum(count_sb_nodes(sb) for sb in ast_node.sbs)
    elif isinstance(ast_node, INPAINT):
        return 1 if ast_node.sb is not None else 0
    elif isinstance(ast_node, AMODAL):
        return (sum(count_sb_nodes(sb) for sb in ast_node.fragment_sbs) +
                sum(count_sb_nodes(sb) for sb in ast_node.amodal_sbs))
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