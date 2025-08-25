"""
Example usage of VoxelDataset with Trainer for GraspGPT training
"""

import sys
import os
# Add parent directories to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
sys.path.extend([current_dir, parent_dir, grandparent_dir])

import torch
from torch.utils.data import DataLoader
from pathlib import Path

from dataset import VoxelDataset, create_dummy_tokenizer
from trainer import Trainer, pad_collate  
from model import graspGPT


def create_example_training_setup():
    """
    Create an example training setup with VoxelDataset and Trainer
    """
    
    # 1. Create dataset with tokenizer function
    data_path = "output/pointclouds/all_voxel_data.pth"
    
    # Replace create_dummy_tokenizer() with your actual tokenizer function
    tokenizer = create_dummy_tokenizer()
    
    dataset = VoxelDataset(
        data_path=data_path,
        tokenizer_fn=tokenizer,
        max_sequence_length=256
    )
    
    print(f"Created dataset with {len(dataset)} samples")
    print(f"Estimated vocab size: {dataset.get_vocab_size()}")
    
    # 2. Create DataLoader with custom collate function
    train_loader = DataLoader(
        dataset=dataset,
        batch_size=8,
        shuffle=True,
        collate_fn=pad_collate,
        num_workers=2,
        pin_memory=True
    )
    
    # 3. Create model configuration
    model_config = graspGPT.get_default_config()
    model_config.vocab_size = dataset.get_vocab_size()
    model_config.block_size = dataset.max_sequence_length
    model_config.model_type = 'gpt-mini'  # Small model for testing
    model_config.use_rope = True  # Use RoPE positioning
    
    print(f"Model config: vocab_size={model_config.vocab_size}, block_size={model_config.block_size}")
    
    # 4. Create model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = graspGPT(model_config)
    model = model.to(device)
    
    # 5. Create trainer configuration  
    trainer_config = Trainer.get_default_config()
    trainer_config.device = device
    trainer_config.batch_size = 8
    trainer_config.learning_rate = 1e-4
    trainer_config.max_iters = 1000
    trainer_config.use_amp = True if device == 'cuda' else False
    trainer_config.scheduler_type = 'cosine'
    trainer_config.warmup_iters = 100
    
    # 6. Create trainer
    trainer = Trainer(trainer_config, model, train_loader)
    
    # 7. Add callback for logging (optional)
    def log_callback(trainer):
        if trainer.iter_num % 50 == 0:
            lr = trainer._get_lr()
            print(f"Iter {trainer.iter_num}: loss={trainer.loss:.4f}, lr={lr:.6f}, dt={trainer.iter_dt:.3f}s")
    
    trainer.add_callback('on_batch_end', log_callback)
    
    return trainer, dataset, model


def test_single_batch():
    """
    Test processing a single batch through the model
    """
    print("\n=== Testing Single Batch ===")
    
    # Create dataset and dataloader
    tokenizer = create_dummy_tokenizer()
    dataset = VoxelDataset("output/pointclouds/all_voxel_data.pth", tokenizer, max_sequence_length=64)
    dataloader = DataLoader(dataset, batch_size=2, collate_fn=pad_collate)
    
    # Create model
    model_config = graspGPT.get_default_config()
    model_config.vocab_size = dataset.get_vocab_size()
    model_config.block_size = dataset.max_sequence_length
    model_config.model_type = 'gpt-nano'  # Very small for testing
    
    model = graspGPT(model_config)
    
    # Test forward pass
    batch = next(iter(dataloader))
    x_batch, y_batch, att_batch = batch
    
    print(f"Batch shapes: x={x_batch.shape}, y={y_batch.shape}, att={att_batch.shape}")
    
    with torch.no_grad():
        logits_heads, loss = model(x_batch, targets=y_batch, attention_mask=att_batch)
        print(f"Forward pass successful!")
        print(f"Logits shape: {logits_heads[0].shape}")
        print(f"Loss: {loss.item() if loss is not None else 'None'}")


if __name__ == "__main__":
    # Test single batch first
    test_single_batch()
    
    # Create full training setup
    print("\n=== Creating Training Setup ===")
    trainer, dataset, model = create_example_training_setup()
    
    print("\nTraining setup created successfully!")
    print("To start training, call: trainer.run()")
    print("\nExample training loop:")
    print("```python")
    print("# Run training for a few iterations")
    print("original_max_iters = trainer.config.max_iters")
    print("trainer.config.max_iters = 10  # Just a few iterations for testing")
    print("trainer.run()")
    print("trainer.config.max_iters = original_max_iters  # Restore original")
    print("```")