import sys
import os
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import random
import argparse
from multiprocessing import Pool, Queue, Manager
from functools import partial

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# 导入所需模块
from graspGPT.model.dataset import VoxelDataset
from graspGPT.model.core import save_voxels
from graspGPT.model.token_manager import get_token_manager


Repeat_times = 1


def convert_to_uint8(data):
    """递归地将数据中的所有tensor和numpy array转换为uint8"""
    if isinstance(data, torch.Tensor):
        return data.to(torch.uint8)
    elif isinstance(data, np.ndarray):
        return data.astype(np.uint8)
    elif isinstance(data, dict):
        return {k: convert_to_uint8(v) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        return type(data)(convert_to_uint8(item) for item in data)
    else:
        return data


# 全局变量存储数据集
global_dataset = None

def init_worker(dataset):
    """初始化工作进程，设置共享的数据集"""
    global global_dataset
    global_dataset = dataset

def process_single_ind(ind):
    """处理单个ind的所有重复"""
    global global_dataset
    
    results = []
    for j in range(Repeat_times):
        voxel_data, scene_grasps, valid_grasp_parampc = global_dataset.prepare_data(ind)
        
        # 转换为uint8
        voxel_data_uint8 = convert_to_uint8(voxel_data)
        scene_grasps_uint8 = convert_to_uint8(scene_grasps)
        valid_grasp_parampc_uint8 = convert_to_uint8(valid_grasp_parampc)

        results.append({
            'voxel_data': voxel_data_uint8,
            'scene_grasps': scene_grasps_uint8,
            'valid_grasp_parampc': valid_grasp_parampc_uint8,
            'ind': ind,
            'repeat': j,
            'volume_dims': global_dataset.volume_dims,
            'bbox_min': global_dataset.bbox_min if hasattr(global_dataset, 'bbox_min') else None,
            'bbox_max': global_dataset.bbox_max if hasattr(global_dataset, 'bbox_max') else None,
            'voxel_size': global_dataset.voxel_size if hasattr(global_dataset, 'voxel_size') else None
        })
    
    return results


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='预计算数据集')
    parser.add_argument('--multiprocess', action='store_true', default=False, 
                        help='启用多进程处理 (默认关闭)')
    parser.add_argument('--num-processes', type=int, default=None,
                        help='指定进程数量 (默认为CPU核心数)')
    args = parser.parse_args()
    
    # 设置数据路径 - 可以是单个文件或目录
    data_path = "output/pointclouds/"  # 使用包含多个batch文件的目录
    # data_path = "output/pointclouds/voxel_data_batch_0.pth"  # 或者单个文件
    
    print("=== 创建VoxelDataset ===")
    
    # 数据集参数
    max_sequence_length = 3512
    grasp_data_path = 'data'
    max_grasps_per_object = 50
    
    try:
        # 创建数据集（只创建一次）
        dataset = VoxelDataset(
            data_path=data_path,
            max_sequence_length=max_sequence_length,
            weights_only=False,
            grasp_data_path=grasp_data_path,
            max_grasps_per_object=max_grasps_per_object
        )
        
        print(f"数据集大小: {len(dataset)} 个样本")
        print(f"词汇表大小: {dataset.get_vocab_size()}")
        print(f"体积维度: {dataset.volume_dims}")
        
        if len(dataset) == 0:
            print("数据集为空，无法进行测试")
            return
            
    except Exception as e:
        print(f"创建数据集失败: {e}")
        return

    # 收集所有结果
    all_results = []
    batch_size = 3000
    batch_count = 0
    total_saved = 0
    
    # 创建保存目录
    output_dir = Path("output/precomputed_data")
    output_dir.mkdir(exist_ok=True)
    
    # 生成随机数用于文件名
    random_id = random.randint(10000, 99999)
    
    # 根据参数决定是否使用多进程
    if args.multiprocess:
        # 使用多进程处理
        if args.num_processes is not None:
            num_processes = args.num_processes
        else:
            num_processes = min(os.cpu_count(), len(dataset))
        print(f"使用 {num_processes} 个进程并行处理")
        
        # 创建初始化函数，传递数据集
        init_func = partial(init_worker, dataset)
        
        with Pool(processes=num_processes, initializer=init_func) as pool:
            # 使用imap显示进度条
            for ind_results in tqdm(pool.imap(process_single_ind, range(len(dataset))), 
                                   total=len(dataset), desc="多进程处理数据集"):
                
                # 将结果添加到总列表
                all_results.extend(ind_results)
                
                # 每3000次保存一个文件
                if len(all_results) >= batch_size:
                    output_path = output_dir / f"precomputed_batch_{random_id}_{batch_count}.pth"
                    torch.save(all_results, output_path)
                    print(f"已保存批次 {batch_count}: {output_path} (包含 {len(all_results)} 个样本)")
                    total_saved += len(all_results)
                    all_results = []  # 清空列表
                    batch_count += 1
    else:
        # 使用单进程处理
        print("使用单进程处理")
        global global_dataset
        global_dataset = dataset  # 设置全局数据集
        
        for ind in tqdm(range(len(dataset)), desc="单进程处理数据集"):
            ind_results = process_single_ind(ind)
            
            # 将结果添加到总列表
            all_results.extend(ind_results)
            
            # 每3000次保存一个文件
            if len(all_results) >= batch_size:
                output_path = output_dir / f"precomputed_batch_{random_id}_{batch_count}.pth"
                torch.save(all_results, output_path)
                print(f"已保存批次 {batch_count}: {output_path} (包含 {len(all_results)} 个样本)")
                total_saved += len(all_results)
                all_results = []  # 清空列表
                batch_count += 1
    
    # 保存剩余的结果
    if all_results:
        output_path = output_dir / f"precomputed_batch_{random_id}_{batch_count}.pth"
        torch.save(all_results, output_path)
        print(f"已保存最后批次 {batch_count}: {output_path} (包含 {len(all_results)} 个样本)")
        total_saved += len(all_results)
    
    print(f"总共保存了 {total_saved} 个样本，分为 {batch_count + 1} 个文件")

if __name__ == "__main__":
    print("Repeat_times =", Repeat_times)
    print("使用方法:")
    print("  单进程模式: python precompute_data.py")
    print("  多进程模式: python precompute_data.py --multiprocess")
    print("  指定进程数: python precompute_data.py --multiprocess --num-processes 4")
    main()