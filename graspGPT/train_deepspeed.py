#!/usr/bin/env python3
"""
DeepSpeed training script for GraspGPT model

This script provides a complete training pipeline for the GraspGPT model using DeepSpeed
for multi-GPU distributed training, including data loading, model configuration, and training loop.

Usage:
    deepspeed --num_gpus=2 train_deepspeed.py [--config CONFIG_FILE] [--data_path DATA_PATH] [--output_dir OUTPUT_DIR]
"""

import os
import sys
import argparse
import time
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler
import wandb
import deepspeed
from deepspeed import comm as dist

# Import local modules
try:
    from model.model import graspGPT
    from model.dataset import VoxelDataset
    from model.trainer import pad_collate
    from model.utils import CfgNode as CN
except ImportError:
    # Handle different import paths
    try:
        from .model.model import graspGPT
        from .model.dataset import VoxelDataset
        from .model.trainer import pad_collate
        from .model.utils import CfgNode as CN
    except ImportError:
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from model.model import graspGPT
        from model.dataset import VoxelDataset
        from model.trainer import pad_collate
        from model.utils import CfgNode as CN


def get_default_config():
    """
    Get default configuration for DeepSpeed training
    
    Returns:
        CN: Configuration node with all training parameters
    """
    C = CN()
    
    # Model configuration
    C.model = graspGPT.get_default_config()
    C.model.model_type = 'gpt-mini'  # or specify custom n_layer, n_head, n_embd
    C.model.vocab_size = None  # Will be set from dataset
    C.model.block_size = 4096   # Maximum sequence length
    C.model.use_rope = True    # Use RoPE position encoding
    C.model.use_flash_attention = True
    
    # Training configuration  
    C.trainer = CN()
    C.trainer.learning_rate = 3e-4
    C.trainer.batch_size = 10  # Global batch size across all GPUs
    C.trainer.micro_batch_size = 2  # Per-GPU micro batch size
    C.trainer.max_iters = 200000
    C.trainer.weight_decay = 0.01
    C.trainer.grad_norm_clip = 2.0
    C.trainer.betas = (0.9, 0.95)
    
    # Learning rate scheduler
    C.trainer.scheduler_type = 'cosine'
    C.trainer.warmup_iters = 500
    C.trainer.min_lr = 1e-6
    
    # Dataset configuration
    C.dataset = CN()
    C.dataset.data_path = "../output/pointclouds/"
    C.dataset.max_sequence_length = 2896
    C.dataset.num_workers = 4
    C.dataset.weights_only = False
    
    # System configuration
    C.system = CN()
    C.system.output_dir = "../output/checkpoints"
    C.system.save_every = 10000  # Save checkpoint every N iterations
    C.system.log_every = 100    # Log progress every N iterations
    C.system.seed = 42
    
    # DeepSpeed configuration
    C.deepspeed = CN()
    C.deepspeed.config_path = "../deepspeed_config.json"
    C.deepspeed.local_rank = -1
    
    # Wandb configuration
    C.wandb = CN()
    C.wandb.enabled = True
    C.wandb.project = 'graspgpt-deepspeed'
    C.wandb.entity = None  # Set to your wandb username/team
    C.wandb.name = None    # Run name, will be auto-generated if None
    C.wandb.tags = ['deepspeed', 'multi-gpu']      # List of tags for the run
    
    return C


def setup_logging(output_dir):
    """Setup logging directory and return log file path"""
    os.makedirs(output_dir, exist_ok=True)
    rank = dist.get_rank() if dist.is_initialized() else 0
    log_file = os.path.join(output_dir, f'training_rank_{rank}.log')
    return log_file


def setup_wandb(config):
    """Initialize Weights & Biases logging (only on rank 0)"""
    if not config.wandb.enabled:
        return
    
    # Only initialize wandb on rank 0
    rank = dist.get_rank() if dist.is_initialized() else 0
    if rank != 0:
        return
    
    # Generate random name if not specified
    run_name = config.wandb.name
    if run_name is None:
        import random
        import string
        random_suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
        run_name = f"graspgpt-deepspeed-{random_suffix}"
    
    wandb.init(
        project=config.wandb.project,
        entity=config.wandb.entity,
        name=run_name,
        tags=config.wandb.tags,
        config={
            'model': config.model,
            'trainer': config.trainer,
            'dataset': config.dataset,
            'system': config.system,
            'deepspeed': config.deepspeed
        },
        resume='allow'  # Allow resuming runs
    )


def log_message(message, log_file=None, print_to_console=True):
    """Log message to file and/or console (only on rank 0 for console)"""
    rank = dist.get_rank() if dist.is_initialized() else 0
    
    if print_to_console and rank == 0:
        print(message)
    
    if log_file:
        with open(log_file, 'a') as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Rank {rank} - {message}\n")


def save_checkpoint(model_engine, config, iter_num, loss, output_dir):
    """Save model checkpoint using DeepSpeed"""
    rank = dist.get_rank() if dist.is_initialized() else 0
    
    if rank == 0:
        checkpoint_dir = os.path.join(output_dir, f'checkpoint_iter_{iter_num}')
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save additional training state
        training_state = {
            'config': config,
            'iter_num': iter_num,
            'loss': loss
        }
        
        training_state_path = os.path.join(checkpoint_dir, 'training_state.json')
        with open(training_state_path, 'w') as f:
            # Convert config to dict for JSON serialization
            config_dict = config.to_dict() if hasattr(config, 'to_dict') else dict(config)
            training_state['config'] = config_dict
            json.dump(training_state, f, indent=2)
    
    # Save DeepSpeed checkpoint
    model_engine.save_checkpoint(output_dir, tag=f'iter_{iter_num}')
    
    # Also save as latest checkpoint
    model_engine.save_checkpoint(output_dir, tag='latest')
    
    return checkpoint_dir if rank == 0 else None


def load_checkpoint(model_engine, checkpoint_dir, config):
    """Load model checkpoint using DeepSpeed"""
    rank = dist.get_rank() if dist.is_initialized() else 0
    
    # Load training state
    training_state_path = os.path.join(checkpoint_dir, 'training_state.json')
    if os.path.exists(training_state_path):
        with open(training_state_path, 'r') as f:
            training_state = json.load(f)
        iter_num = training_state.get('iter_num', 0)
        loss = training_state.get('loss', float('inf'))
    else:
        iter_num = 0
        loss = float('inf')
    
    # Load DeepSpeed checkpoint
    _, client_state = model_engine.load_checkpoint(checkpoint_dir)
    
    if rank == 0:
        print(f"Loaded checkpoint from {checkpoint_dir}, iteration {iter_num}, loss {loss}")
    
    return iter_num, loss


def create_model_and_dataset(config):
    """Create model and dataset based on configuration"""
    rank = dist.get_rank() if dist.is_initialized() else 0
    
    if rank == 0:
        print("Setting up dataset...")
    
    dataset = VoxelDataset(
        data_path=config.dataset.data_path,
        max_sequence_length=config.dataset.max_sequence_length,
        weights_only=config.dataset.weights_only
    )
    
    if rank == 0:
        print(f"Dataset loaded: {len(dataset)} samples")
    
    # Get vocabulary size from dataset
    vocab_size = dataset.get_vocab_size()
    if rank == 0:
        print(f"Vocabulary size: {vocab_size}")
    
    # Update model config with vocab size
    config.model.vocab_size = vocab_size
    config.model.block_size = config.dataset.max_sequence_length
    
    if rank == 0:
        print("Creating model...")
    
    model = graspGPT(config.model)
    
    if rank == 0:
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    return model, dataset


def create_dataloader(dataset, config):
    """Create distributed data loader"""
    # Create distributed sampler
    sampler = DistributedSampler(dataset, shuffle=True) if dist.is_initialized() else None
    
    return DataLoader(
        dataset,
        batch_size=config.trainer.micro_batch_size,
        shuffle=(sampler is None),  # Don't shuffle if using distributed sampler
        sampler=sampler,
        num_workers=config.dataset.num_workers,
        collate_fn=pad_collate,
        pin_memory=True,
        drop_last=True
    ), sampler


def create_deepspeed_config(config):
    """Create DeepSpeed configuration dictionary"""
    # Load base config from JSON file
    with open(config.deepspeed.config_path, 'r') as f:
        ds_config = json.load(f)
    
    # Override with training configuration
    ds_config.update({
        "train_batch_size": config.trainer.batch_size,
        "train_micro_batch_size_per_gpu": config.trainer.micro_batch_size,
        "gradient_accumulation_steps": config.trainer.batch_size // (config.trainer.micro_batch_size * dist.get_world_size()) if dist.is_initialized() else 1,
        "gradient_clipping": config.trainer.grad_norm_clip,
    })
    
    # Update optimizer settings
    if "optimizer" in ds_config:
        ds_config["optimizer"]["params"].update({
            "lr": config.trainer.learning_rate,
            "betas": config.trainer.betas,
            "weight_decay": config.trainer.weight_decay
        })
    
    # Update scheduler settings
    if "scheduler" in ds_config:
        ds_config["scheduler"]["params"].update({
            "warmup_min_lr": config.trainer.min_lr,
            "warmup_max_lr": config.trainer.learning_rate,
            "warmup_num_steps": config.trainer.warmup_iters,
            "total_num_steps": config.trainer.max_iters
        })
    
    return ds_config


def training_step(model_engine, batch, config):
    """Single training step"""
    x, y, att = batch
    
    # Move to device (DeepSpeed handles this automatically)
    x = x.to(model_engine.local_rank, non_blocking=True)
    y = y.to(model_engine.local_rank, non_blocking=True)
    if att is not None:
        att = att.to(model_engine.local_rank, non_blocking=True)
    
    # Forward pass
    logits, loss = model_engine(x, targets=y, attention_mask=att)
    
    # Backward pass
    model_engine.backward(loss)
    
    # Optimizer step
    model_engine.step()
    
    return loss


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train GraspGPT model with DeepSpeed')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config file (JSON format)')
    parser.add_argument('--data_path', type=str, default=None,
                       help='Path to training data (overrides config)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for checkpoints (overrides config)')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint directory to resume training from')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Global batch size (overrides config)')
    parser.add_argument('--micro_batch_size', type=int, default=None,
                       help='Micro batch size per GPU (overrides config)')
    parser.add_argument('--learning_rate', type=float, default=None,
                       help='Learning rate (overrides config)')
    parser.add_argument('--max_iters', type=int, default=None,
                       help='Maximum iterations (overrides config)')
    parser.add_argument('--no_wandb', action='store_true',
                       help='Disable wandb logging')
    parser.add_argument('--wandb_project', type=str, default=None,
                       help='Wandb project name (overrides config)')
    parser.add_argument('--wandb_name', type=str, default=None,
                       help='Wandb run name (overrides config)')
    parser.add_argument('--local_rank', type=int, default=-1,
                       help='Local rank for distributed training')
    parser.add_argument('--deepspeed_config', type=str, default=None,
                       help='Path to DeepSpeed config file (overrides config)')
    
    # Parse known args to handle DeepSpeed arguments
    args, unknown_args = parser.parse_known_args()
    
    # Initialize DeepSpeed distributed training
    deepspeed.init_distributed()
    
    # Load configuration
    config = get_default_config()
    
    if args.config:
        rank = dist.get_rank()
        if rank == 0:
            print(f"Loading config from {args.config}")
        with open(args.config, 'r') as f:
            config_dict = json.load(f)
        config.merge_from_dict(config_dict)
    
    # Override config with command line arguments
    if args.data_path:
        config.dataset.data_path = args.data_path
    if args.output_dir:
        config.system.output_dir = args.output_dir
    if args.batch_size:
        config.trainer.batch_size = args.batch_size
    if args.micro_batch_size:
        config.trainer.micro_batch_size = args.micro_batch_size
    if args.learning_rate:
        config.trainer.learning_rate = args.learning_rate
    if args.max_iters:
        config.trainer.max_iters = args.max_iters
    if args.no_wandb:
        config.wandb.enabled = False
    if args.wandb_project:
        config.wandb.project = args.wandb_project
    if args.wandb_name:
        config.wandb.name = args.wandb_name
    if args.deepspeed_config:
        config.deepspeed.config_path = args.deepspeed_config
    
    config.deepspeed.local_rank = args.local_rank
    
    # Set random seed for reproducibility
    torch.manual_seed(config.system.seed + dist.get_rank())
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.system.seed + dist.get_rank())
    
    # Setup logging
    log_file = setup_logging(config.system.output_dir)
    rank = dist.get_rank()
    log_message(f"Starting GraspGPT DeepSpeed training on rank {rank}", log_file)
    log_message(f"World size: {dist.get_world_size()}", log_file)
    
    # Setup wandb (only on rank 0)
    setup_wandb(config)
    
    try:
        # Create model and dataset
        model, dataset = create_model_and_dataset(config)
        
        # Create distributed data loader
        train_loader, train_sampler = create_dataloader(dataset, config)
        log_message(f"DataLoader created with micro_batch_size={config.trainer.micro_batch_size}", log_file)
        
        # Create DeepSpeed configuration
        ds_config = create_deepspeed_config(config)
        
        # Initialize DeepSpeed engine
        model_engine, optimizer, train_loader, lr_scheduler = deepspeed.initialize(
            model=model,
            config=ds_config,
            training_data=dataset,
            collate_fn=pad_collate
        )
        
        if rank == 0:
            print(f"DeepSpeed engine initialized")
            print(f"Effective batch size: {model_engine.train_batch_size()}")
            print(f"Gradient accumulation steps: {model_engine.gradient_accumulation_steps()}")
        
        # Resume from checkpoint if specified
        start_iter = 0
        if args.resume:
            if rank == 0:
                print(f"Resuming training from {args.resume}")
            start_iter, loss = load_checkpoint(model_engine, args.resume, config)
            log_message(f"Resumed training from iteration {start_iter}, loss: {loss:.4f}", log_file)
        
        # Start training
        log_message("Starting training loop", log_file)
        if rank == 0:
            print("Training started...")
        
        model_engine.train()
        iter_num = start_iter
        data_iter = iter(train_loader)
        
        while iter_num < config.trainer.max_iters:
            try:
                batch = next(data_iter)
            except StopIteration:
                # Set epoch for distributed sampler
                if train_sampler is not None:
                    train_sampler.set_epoch(iter_num // len(train_loader))
                data_iter = iter(train_loader)
                batch = next(data_iter)
            
            # Training step
            iter_start_time = time.time()
            loss = training_step(model_engine, batch, config)
            iter_time = time.time() - iter_start_time
            
            # Logging
            if iter_num % config.system.log_every == 0:
                lr = model_engine.get_lr()[0]
                loss_value = loss.item()
                message = (f"Iter {iter_num:6d} | "
                          f"Loss: {loss_value:.4f} | "
                          f"LR: {lr:.2e} | "
                          f"Time: {iter_time*1000:.1f}ms")
                log_message(message, log_file)
                
                # Log to wandb (only on rank 0)
                if config.wandb.enabled and wandb.run is not None and rank == 0:
                    wandb.log({
                        'train/loss': loss_value,
                        'train/learning_rate': lr,
                        'train/iter_time_ms': iter_time * 1000,
                        'iteration': iter_num
                    }, step=iter_num)
            
            # Checkpointing
            if iter_num % config.system.save_every == 0 and iter_num > 0:
                if rank == 0:
                    print(f"Saving checkpoint at iteration {iter_num}")
                checkpoint_path = save_checkpoint(
                    model_engine,
                    config,
                    iter_num,
                    loss.item(),
                    config.system.output_dir
                )
                if checkpoint_path:
                    log_message(f"Checkpoint saved: {checkpoint_path}", log_file)
            
            iter_num += 1
        
        # Save final checkpoint
        if rank == 0:
            print(f"Training completed: reached max_iters {config.trainer.max_iters}")
        log_message(f"Training completed at iteration {iter_num}", log_file)
        
        final_path = save_checkpoint(
            model_engine,
            config,
            iter_num,
            loss.item(),
            config.system.output_dir
        )
        if final_path:
            log_message(f"Final checkpoint saved: {final_path}", log_file)
    
    except KeyboardInterrupt:
        if rank == 0:
            print("\nTraining interrupted by user")
        log_message("Training interrupted by user", log_file)
        # Save checkpoint before exit
        if 'model_engine' in locals():
            checkpoint_path = save_checkpoint(
                model_engine,
                config,
                iter_num,
                loss.item() if 'loss' in locals() else float('inf'),
                config.system.output_dir
            )
            if checkpoint_path and rank == 0:
                print(f"Interrupt checkpoint saved: {checkpoint_path}")
        if config.wandb.enabled and wandb.run is not None and rank == 0:
            wandb.finish()
    except Exception as e:
        error_msg = f"Training failed with error: {str(e)}"
        if rank == 0:
            print(error_msg)
        log_message(error_msg, log_file)
        raise
    
    if rank == 0:
        print("Training script completed")
    log_message("Training script completed", log_file)
    
    # Close wandb run (only on rank 0)
    if config.wandb.enabled and wandb.run is not None and rank == 0:
        wandb.finish()


if __name__ == "__main__":
    main()