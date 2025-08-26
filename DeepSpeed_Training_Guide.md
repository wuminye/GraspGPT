# GraspGPT DeepSpeed Training Guide

本指南介绍如何使用DeepSpeed框架进行GraspGPT模型的多卡分布式训练。

## 安装依赖

首先确保安装了所需的依赖包：

```bash

# 安装DeepSpeed
pip install deepspeed

# 安装其他可能需要的依赖
pip install transformers accelerate
```

## 文件结构

- `deepspeed_config.json`: DeepSpeed配置文件
- `graspGPT/train_deepspeed.py`: 支持DeepSpeed的训练脚本
- `run_deepspeed_training.sh`: 便捷的启动脚本
- `DeepSpeed_Training_Guide.md`: 本指南文件

## 配置说明

### DeepSpeed配置 (deepspeed_config.json)

主要配置项：
- `zero_optimization.stage`: ZeRO优化阶段 (1, 2, 3)
- `train_batch_size`: 全局批次大小
- `train_micro_batch_size_per_gpu`: 每GPU的微批次大小
- `gradient_accumulation_steps`: 梯度累积步数
- `fp16/bf16`: 混合精度训练设置
- `optimizer`: 优化器配置
- `scheduler`: 学习率调度器配置

### 训练配置

在 `train_deepspeed.py` 中的主要配置：
- `config.trainer.batch_size`: 全局批次大小
- `config.trainer.micro_batch_size`: 每GPU微批次大小
- `config.trainer.learning_rate`: 学习率
- `config.trainer.max_iters`: 最大训练步数

## 使用方法

### 方法1: 使用便捷脚本

```bash
# 使用2块GPU训练
./run_deepspeed_training.sh 2

# 使用4块GPU训练
./run_deepspeed_training.sh 4

# 使用自定义参数
./run_deepspeed_training.sh 2 --batch_size 40 --learning_rate 5e-4
```

### 方法2: 直接使用DeepSpeed命令

```bash
cd graspGPT

# 基本用法 - 2块GPU
deepspeed --num_gpus=2 train_deepspeed.py \
    --deepspeed_config ../deepspeed_config.json \
    --batch_size 20 \
    --micro_batch_size 5

# 使用特定GPU
CUDA_VISIBLE_DEVICES=0,1 deepspeed --num_gpus=2 train_deepspeed.py \
    --deepspeed_config ../deepspeed_config.json

# 从检查点恢复训练
deepspeed --num_gpus=2 train_deepspeed.py \
    --deepspeed_config ../deepspeed_config.json \
    --resume ../output/checkpoints/checkpoint_iter_10000

# 自定义数据路径和输出目录
deepspeed --num_gpus=2 train_deepspeed.py \
    --deepspeed_config ../deepspeed_config.json \
    --data_path /path/to/your/data \
    --output_dir /path/to/output
```

### 方法3: 使用hostfile进行多节点训练

创建hostfile文件：
```
node1 slots=4
node2 slots=4
```

运行命令：
```bash
deepspeed --hostfile hostfile train_deepspeed.py \
    --deepspeed_config ../deepspeed_config.json
```

## 命令行参数

训练脚本支持以下主要参数：

- `--config`: 自定义配置文件路径
- `--data_path`: 训练数据路径
- `--output_dir`: 输出目录
- `--resume`: 检查点目录路径（用于恢复训练）
- `--batch_size`: 全局批次大小
- `--micro_batch_size`: 每GPU微批次大小
- `--learning_rate`: 学习率
- `--max_iters`: 最大训练步数
- `--no_wandb`: 禁用wandb日志
- `--wandb_project`: wandb项目名称
- `--deepspeed_config`: DeepSpeed配置文件路径

## 性能优化建议

### 1. 批次大小设置（自动计算）

训练脚本会自动计算正确的批次大小参数：

```python
# DeepSpeed要求：
# train_batch_size = micro_batch_size × gradient_accumulation_steps × world_size

# 示例：使用2块GPU，micro_batch_size=8，期望batch_size=32
# gradient_accumulation_steps = 32 / (8 × 2) = 2
# 实际train_batch_size = 8 × 2 × 2 = 32 ✓
```

**参数说明**：
- `micro_batch_size`: 每个GPU的批次大小（根据GPU内存调整）
- `batch_size`: 期望的全局批次大小（训练脚本会自动调整到最接近的可行值）
- `gradient_accumulation_steps`: 自动计算
- `world_size`: GPU数量（自动获取）

### 2. ZeRO优化阶段选择

- **Stage 1**: 优化器状态分片，内存节省较少
- **Stage 2**: 优化器状态+梯度分片，推荐用于中等大小模型
- **Stage 3**: 优化器状态+梯度+参数分片，适合大模型

### 3. 内存优化

```json
{
  "zero_optimization": {
    "stage": 2,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    },
    "offload_param": {
      "device": "cpu", 
      "pin_memory": true
    }
  }
}
```

### 4. 混合精度训练

```json
{
  "fp16": {
    "enabled": true,
    "loss_scale": 0,
    "initial_scale_power": 16
  }
}
```

## 监控和调试

### 1. Wandb集成

训练过程会自动记录到Wandb：
- 训练损失
- 学习率变化
- 迭代时间
- GPU利用率

### 2. 日志文件

每个进程会生成独立的日志文件：
- `training_rank_0.log`
- `training_rank_1.log`
- ...

### 3. 检查点管理

检查点自动保存到 `output_dir` 目录：
```
output_dir/
├── checkpoint_iter_1000/
├── checkpoint_iter_2000/
└── latest/
```

## 常见问题

### 1. CUDA内存不足

- 减少 `micro_batch_size`
- 启用CPU offloading
- 使用ZeRO Stage 3

### 2. 通信错误

- 检查NCCL版本
- 设置正确的网络接口
- 确保所有节点版本一致

### 3. 性能问题

- 调整 `gradient_accumulation_steps`
- 启用activation checkpointing
- 优化数据加载器的 `num_workers`

## 与原始训练的区别

| 特性 | 原始训练 | DeepSpeed训练 |
|------|----------|---------------|
| GPU支持 | 单GPU | 多GPU分布式 |
| 内存优化 | 基础 | ZeRO优化 |
| 混合精度 | 可选 | 自动优化 |
| 检查点 | 简单 | 分布式保存 |
| 监控 | 基础日志 | 详细指标 |

## 预期性能提升

使用2块GPU的预期提升：
- **训练速度**: 约1.7-1.9倍
- **内存使用**: 节省30-50%
- **模型并行**: 支持更大模型

使用4块GPU的预期提升：
- **训练速度**: 约3.2-3.6倍  
- **内存使用**: 节省50-70%
- **扩展性**: 线性扩展能力

## 示例配置

### 小模型 (gpt-mini)
```bash
deepspeed --num_gpus=2 train_deepspeed.py \
    --batch_size 20 \
    --micro_batch_size 5 \
    --learning_rate 3e-4
```

### 中等模型 (gpt-medium)
```bash
deepspeed --num_gpus=4 train_deepspeed.py \
    --batch_size 16 \
    --micro_batch_size 2 \
    --learning_rate 2e-4
```

### 大模型配置
修改 `deepspeed_config.json`：
```json
{
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {"device": "cpu"},
    "offload_param": {"device": "cpu"}
  }
}
```

这样就完成了GraspGPT模型的DeepSpeed多卡训练框架改写，保持了原有的代码结构和功能特性。