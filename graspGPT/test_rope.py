#!/usr/bin/env python3
"""
Test script for RoPE implementation in GraspGPT
"""

import torch
import sys
import os

# Add the project root to the path
sys.path.append('/home/wuminye/gitcode/GraspGPT')

from model.model import graspGPT


def test_rope_model():
    """Test basic RoPE model functionality"""
    print("Testing RoPE implementation...")
    
    # Get default config and enable RoPE
    config = graspGPT.get_default_config()
    config.model_type = None  # Use specific parameters instead
    config.vocab_size = 1000
    config.block_size = 64
    config.n_layer = 2
    config.n_head = 4
    config.n_embd = 128
    config.use_rope = True  # Enable RoPE
    
    # Create model
    print("\n1. Creating RoPE model...")
    model = graspGPT(config)
    
    # Create test input
    batch_size = 2
    seq_len = 16
    num_features = 1
    
    # Test input format: (batch_size, seq_len, num_features)
    idx = torch.randint(0, config.vocab_size, (batch_size, seq_len, num_features))
    targets = torch.randint(0, config.vocab_size, (batch_size, seq_len, num_features))
    
    print(f"Input shape: {idx.shape}")
    print(f"Targets shape: {targets.shape}")
    
    # Test forward pass
    print("\n2. Testing forward pass...")
    model.train()
    logits_heads, loss = model(idx, targets=targets)
    
    print(f"Logits shape: {logits_heads[0].shape}")
    print(f"Loss: {loss.item():.4f}")
    
    # Test generation
    print("\n3. Testing generation...")
    model.eval()
    with torch.no_grad():
        # Start with a short sequence
        start_idx = idx[:1, :8, :]  # First sample, first 8 tokens
        print(f"Start sequence shape: {start_idx.shape}")
        
        generated = model.generate(
            start_idx, 
            max_new_tokens=8,
            temperature=1.0,
            do_sample=True,
            top_k=10
        )
        
        print(f"Generated sequence shape: {generated.shape}")
        print(f"Generated tokens: {generated[0, :, 0].tolist()}")
    
    print("\n✅ RoPE model test completed successfully!")
    
    
def test_traditional_model():
    """Test traditional absolute position encoding model"""
    print("\n" + "="*50)
    print("Testing Traditional Absolute Position Encoding...")
    
    # Get default config and disable RoPE
    config = graspGPT.get_default_config()
    config.model_type = None  # Use specific parameters instead
    config.vocab_size = 1000
    config.block_size = 64
    config.n_layer = 2
    config.n_head = 4
    config.n_embd = 128
    config.use_rope = False  # Disable RoPE
    
    # Create model
    print("\n1. Creating traditional model...")
    model = graspGPT(config)
    
    # Create test input
    batch_size = 2
    seq_len = 16
    num_features = 3
    
    idx = torch.randint(0, config.vocab_size, (batch_size, seq_len, num_features))
    targets = torch.randint(0, config.vocab_size, (batch_size, seq_len, num_features))
    
    # Test forward pass
    print("\n2. Testing forward pass...")
    model.train()
    logits_heads, loss = model(idx, targets=targets)
    
    print(f"Logits shape: {logits_heads[0].shape}")
    print(f"Loss: {loss.item():.4f}")
    
    print("\n✅ Traditional model test completed successfully!")


def compare_models():
    """Compare RoPE vs Traditional models"""
    print("\n" + "="*50)
    print("Comparing RoPE vs Traditional Models...")
    
    # Common config
    base_config = graspGPT.get_default_config()
    base_config.model_type = None  # Use specific parameters instead
    base_config.vocab_size = 1000
    base_config.block_size = 64
    base_config.n_layer = 2
    base_config.n_head = 4
    base_config.n_embd = 128
    
    # Create both models
    rope_config = graspGPT.get_default_config()
    rope_config.model_type = None
    rope_config.vocab_size = base_config.vocab_size
    rope_config.block_size = base_config.block_size
    rope_config.n_layer = base_config.n_layer
    rope_config.n_head = base_config.n_head
    rope_config.n_embd = base_config.n_embd
    rope_config.use_rope = True
    
    traditional_config = graspGPT.get_default_config()
    traditional_config.model_type = None
    traditional_config.vocab_size = base_config.vocab_size
    traditional_config.block_size = base_config.block_size
    traditional_config.n_layer = base_config.n_layer
    traditional_config.n_head = base_config.n_head
    traditional_config.n_embd = base_config.n_embd
    traditional_config.use_rope = False
    
    print("\nCreating both models...")
    rope_model = graspGPT(rope_config)
    traditional_model = graspGPT(traditional_config)
    
    # Compare parameter counts
    rope_params = sum(p.numel() for p in rope_model.parameters())
    traditional_params = sum(p.numel() for p in traditional_model.parameters())
    
    print(f"\nParameter comparison:")
    print(f"RoPE model parameters: {rope_params:,}")
    print(f"Traditional model parameters: {traditional_params:,}")
    print(f"Difference: {abs(rope_params - traditional_params):,}")
    
    print("\n✅ Model comparison completed!")


if __name__ == "__main__":
    print("GraspGPT RoPE Implementation Test")
    print("="*50)
    
    try:
        # Test RoPE implementation
        test_rope_model()
        
        # Test traditional implementation
        test_traditional_model()
        
        # Compare models
        compare_models()
        
        print("\n🎉 All tests passed successfully!")
        print("The RoPE position encoding has been successfully integrated into GraspGPT!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()