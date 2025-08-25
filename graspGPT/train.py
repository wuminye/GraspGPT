#!/usr/bin/env python3
"""
Training script for GraspGPT model

This script provides a complete training pipeline for the GraspGPT model,
including data loading, model configuration, and training loop with callbacks.

Usage:
    python train.py [--config CONFIG_FILE] [--data_path DATA_PATH] [--output_dir OUTPUT_DIR]
"""

import os
import sys
import argparse
import time
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb

# Import local modules
try:
    from model.model import graspGPT
    from model.dataset import VoxelDataset
    from model.trainer import Trainer, pad_collate
    from model.utils import CfgNode as CN
except ImportError:
    # Handle different import paths
    try:
        from .model.model import graspGPT
        from .model.dataset import VoxelDataset
        from .model.trainer import Trainer, pad_collate
        from .model.utils import CfgNode as CN
    except ImportError:
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from model.model import graspGPT
        from model.dataset import VoxelDataset
        from model.trainer import Trainer, pad_collate
        from model.utils import CfgNode as CN


def get_default_config():
    """
    Get default configuration for training
    
    Returns:
        CN: Configuration node with all training parameters
    """
    C = CN()
    
    # Model configuration
    C.model = graspGPT.get_default_config()
    C.model.model_type = 'gpt-mini'  # or specify custom n_layer, n_head, n_embd
    C.model.vocab_size = None  # Will be set from dataset
    C.model.block_size = 4096   # Maximum sequence length
    C.model.use_rope = False    # Use RoPE position encoding
    C.model.use_flash_attention = True
    
    # Training configuration  
    C.trainer = Trainer.get_default_config()
    C.trainer.learning_rate = 3e-4
    C.trainer.batch_size = 9
    C.trainer.max_iters = 100000
    C.trainer.weight_decay = 0.01
    C.trainer.grad_norm_clip = 2.0
    C.trainer.use_amp = False  # Mixed precision training
    
    # Learning rate scheduler
    C.trainer.scheduler_type = 'cosine'
    C.trainer.warmup_iters = 500
    C.trainer.min_lr = 1e-6
    
    # Dataset configuration
    C.dataset = CN()
    C.dataset.data_path = "../output/pointclouds/all_voxel_data.pth"
    C.dataset.max_sequence_length = 2096
    C.dataset.num_workers = 4
    C.dataset.weights_only = False
    
    # System configuration
    C.system = CN()
    C.system.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    C.system.output_dir = "../output/checkpoints"
    C.system.save_every = 10000  # Save checkpoint every N iterations
    C.system.log_every = 100    # Log progress every N iterations
    C.system.seed = 42
    
    # Wandb configuration
    C.wandb = CN()
    C.wandb.enabled = True
    C.wandb.project = 'graspgpt'
    C.wandb.entity = None  # Set to your wandb username/team
    C.wandb.name = None    # Run name, will be auto-generated if None
    C.wandb.tags = []      # List of tags for the run
    
    return C


def setup_logging(output_dir):
    """Setup logging directory and return log file path"""
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, 'training.log')
    return log_file


def setup_wandb(config):
    """Initialize Weights & Biases logging"""
    if not config.wandb.enabled:
        return
    
    # Generate random name if not specified
    run_name = config.wandb.name
    if run_name is None:
        import random
        import string
        random_suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
        run_name = f"graspgpt-{random_suffix}"
    
    wandb.init(
        project=config.wandb.project,
        entity=config.wandb.entity,
        name=run_name,
        tags=config.wandb.tags,
        config={
            'model': config.model,
            'trainer': config.trainer,
            'dataset': config.dataset,
            'system': config.system
        },
        resume='allow'  # Allow resuming runs
    )


def log_message(message, log_file=None, print_to_console=True):
    """Log message to file and/or console"""
    if print_to_console:
        print(message)
    
    if log_file:
        with open(log_file, 'a') as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")


def save_checkpoint(model, optimizer, scheduler, config, iter_num, loss, output_dir):
    """Save model checkpoint"""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'config': config,
        'iter_num': iter_num,
        'loss': loss
    }
    
    checkpoint_path = os.path.join(output_dir, f'checkpoint_iter_{iter_num}.pt')
    torch.save(checkpoint, checkpoint_path)
    
    # Also save as latest checkpoint
    latest_path = os.path.join(output_dir, 'checkpoint_latest.pt')
    torch.save(checkpoint, latest_path)
    
    return checkpoint_path


def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None):
    """Load model checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    iter_num = checkpoint.get('iter_num', 0)
    loss = checkpoint.get('loss', float('inf'))
    
    return iter_num, loss


def create_model_and_dataset(config):
    """Create model and dataset based on configuration"""
    print("Setting up dataset...")
    dataset = VoxelDataset(
        data_path=config.dataset.data_path,
        max_sequence_length=config.dataset.max_sequence_length,
        weights_only=config.dataset.weights_only
    )
    
    print(f"Dataset loaded: {len(dataset)} samples")
    
    # Get vocabulary size from dataset
    vocab_size = dataset.get_vocab_size()
    print(f"Vocabulary size: {vocab_size}")
    
    # Update model config with vocab size
    config.model.vocab_size = vocab_size
    config.model.block_size = config.dataset.max_sequence_length
    
    print("Creating model...")
    model = graspGPT(config.model)
    
    # Move model to device
    device = torch.device(config.system.device)
    model = model.to(device)
    
    print(f"Model moved to device: {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    return model, dataset


def create_dataloader(dataset, config):
    """Create data loader with appropriate configuration"""
    return DataLoader(
        dataset,
        batch_size=config.trainer.batch_size,
        shuffle=True,
        num_workers=config.dataset.num_workers,
        collate_fn=pad_collate,
        pin_memory=torch.cuda.is_available(),
        drop_last=True
    )


def create_callbacks(config, log_file):
    """Create training callbacks for logging and checkpointing"""
    output_dir = config.system.output_dir
    
    def on_batch_end_callback(trainer):
        """Callback executed after each training batch"""
        # Logging
        if trainer.iter_num % config.system.log_every == 0:
            lr = trainer._get_lr()
            loss_value = trainer.loss.item()
            message = (f"Iter {trainer.iter_num:6d} | "
                      f"Loss: {loss_value:.4f} | "
                      f"LR: {lr:.2e} | "
                      f"Time: {trainer.iter_dt*1000:.1f}ms")
            log_message(message, log_file)
            
            # Log to wandb
            if config.wandb.enabled and wandb.run is not None:
                wandb.log({
                    'train/loss': loss_value,
                    'train/learning_rate': lr,
                    'train/iter_time_ms': trainer.iter_dt * 1000,
                    'iteration': trainer.iter_num
                }, step=trainer.iter_num)
        
        # Checkpointing
        if trainer.iter_num % config.system.save_every == 0:
            print(f"Saving checkpoint at iteration {trainer.iter_num}")
            checkpoint_path = save_checkpoint(
                trainer.model,
                trainer.optimizer, 
                trainer.scheduler,
                config,
                trainer.iter_num,
                trainer.loss.item(),
                output_dir
            )
            log_message(f"Checkpoint saved: {checkpoint_path}", log_file)
        
        # Stop condition
        if config.trainer.max_iters and trainer.iter_num >= config.trainer.max_iters:
            print(f"Training completed: reached max_iters {config.trainer.max_iters}")
            log_message(f"Training completed at iteration {trainer.iter_num}", log_file)
            
            # Save final checkpoint
            final_path = save_checkpoint(
                trainer.model,
                trainer.optimizer,
                trainer.scheduler, 
                config,
                trainer.iter_num,
                trainer.loss.item(),
                output_dir
            )
            log_message(f"Final checkpoint saved: {final_path}", log_file)
            
            # Exit training loop
            raise StopIteration("Training completed")
    
    return on_batch_end_callback


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train GraspGPT model')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config file (JSON format)')
    parser.add_argument('--data_path', type=str, default=None,
                       help='Path to training data (overrides config)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for checkpoints (overrides config)')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume training from')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Batch size (overrides config)')
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
    
    args = parser.parse_args()
    
    # Load configuration
    config = get_default_config()
    
    if args.config:
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
    
    # Set random seed for reproducibility
    torch.manual_seed(config.system.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.system.seed)
    
    # Setup logging
    log_file = setup_logging(config.system.output_dir)
    log_message("Starting GraspGPT training", log_file)
    log_message(f"Config: {config}", log_file)
    
    # Setup wandb
    setup_wandb(config)
    
    try:
        # Create model and dataset
        model, dataset = create_model_and_dataset(config)
        
        # Create data loader
        train_loader = create_dataloader(dataset, config)
        log_message(f"DataLoader created with batch_size={config.trainer.batch_size}", log_file)
        
        # Create trainer
        trainer = Trainer(config.trainer, model, train_loader)
        
        # Add callbacks
        callback = create_callbacks(config, log_file)
        trainer.add_callback('on_batch_end', callback)
        
        # Resume from checkpoint if specified
        if args.resume:
            print(f"Resuming training from {args.resume}")
            iter_num, loss = load_checkpoint(args.resume, model, trainer.optimizer, trainer.scheduler)
            trainer.iter_num = iter_num
            log_message(f"Resumed training from iteration {iter_num}, loss: {loss:.4f}", log_file)
        
        # Start training
        log_message("Starting training loop", log_file)
        print("Training started...")
        trainer.run()
        
    except StopIteration as e:
        print(f"Training stopped: {e}")
        log_message(f"Training stopped: {e}", log_file)
        if config.wandb.enabled and wandb.run is not None:
            wandb.finish()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        log_message("Training interrupted by user", log_file)
        # Save checkpoint before exit
        if 'trainer' in locals():
            checkpoint_path = save_checkpoint(
                trainer.model,
                trainer.optimizer,
                trainer.scheduler,
                config,
                trainer.iter_num,
                trainer.loss.item() if hasattr(trainer, 'loss') else float('inf'),
                config.system.output_dir
            )
            print(f"Interrupt checkpoint saved: {checkpoint_path}")
        if config.wandb.enabled and wandb.run is not None:
            wandb.finish()
    except Exception as e:
        error_msg = f"Training failed with error: {str(e)}"
        print(error_msg)
        log_message(error_msg, log_file)
        raise
    
    print("Training script completed")
    log_message("Training script completed", log_file)
    
    # Close wandb run
    if config.wandb.enabled and wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()