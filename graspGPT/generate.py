#!/usr/bin/env python3
"""
Standard PyTorch model generation script for GraspGPT (without DeepSpeed)

This script loads a model trained with DeepSpeed and generates sequences using standard PyTorch.
It supports single-GPU generation with simplified checkpoint loading.

使用方法：

  1. 通过命令行直接提供 token IDs（逗号分隔）：

  python graspGPT/generate.py \
    --checkpoint_path /path/to/checkpoints/iter_10000 \
    --prompt_tokens "1,2,3,4,5" \
    --max_new_tokens 50 \
    --do_sample

  2. 通过文件提供 token IDs（逗号分隔或每行一个）：

  python graspGPT/generate.py \
    --checkpoint_path /path/to/checkpoints/iter_10000 \
    --prompt_file example_prompt.txt \
    --temperature 0.8

  3. 无条件生成（不提供 prompt）：

  python graspGPT/generate.py \
    --checkpoint_path /path/to/checkpoints/iter_10000 \
    --num_sequences 3

注意：
- prompt 必须是 token ID 序列，不是文本字符串
- 支持从训练数据中提取的实际 token sequence
- 可以指定特定的形状、坐标或命令 token 组合
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn

# Import local modules
try:
    from model.model import graspGPT
    from model.utils import CfgNode as CN
    from model.parser_and_serializer import Serializer, Parser
except ImportError:
    # Handle different import paths
    try:
        from .model.model import graspGPT
        from .model.utils import CfgNode as CN
        from .model.parser_and_serializer import serialize_sequence, deserialize_sequence
    except ImportError:
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from model.model import graspGPT
        from model.utils import CfgNode as CN
        from model.parser_and_serializer import Serializer, Parser


def parse_config_string(config_str):
    """Parse string-formatted configuration back to dictionary"""
    config_dict = {}
    for line in config_str.strip().split('\n'):
        if ':' in line:
            key, value = line.split(':', 1)
            key = key.strip()
            value = value.strip()
            
            # Try to parse the value
            try:
                # Handle boolean values
                if value.lower() == 'true':
                    value = True
                elif value.lower() == 'false':
                    value = False
                elif value.lower() == 'none':
                    value = None
                # Handle tuples
                elif value.startswith('(') and value.endswith(')'):
                    value = eval(value)
                # Handle lists
                elif value.startswith('[') and value.endswith(']'):
                    value = eval(value)
                # Try to parse as number
                elif '.' in value:
                    try:
                        value = float(value)
                    except ValueError:
                        pass  # Keep as string
                else:
                    try:
                        value = int(value)
                    except ValueError:
                        pass  # Keep as string
            except:
                pass  # Keep as string if parsing fails
            
            config_dict[key] = value
    
    return config_dict


def load_training_config(checkpoint_dir):
    """Load training configuration from checkpoint directory"""
    config_file = None
    
    # First try to find training_state.json in the checkpoint directory
    training_state_path = os.path.join(checkpoint_dir, 'training_state.json')
    if os.path.exists(training_state_path):
        with open(training_state_path, 'r') as f:
            training_state = json.load(f)
            if 'config' in training_state:
                config_data = training_state['config']
                
                # Check if config is in string format (from training script)
                if isinstance(config_data, dict):
                    # Handle case where each section is a string
                    parsed_config = {}
                    for section_name, section_value in config_data.items():
                        if isinstance(section_value, str):
                            parsed_config[section_name] = parse_config_string(section_value)
                        else:
                            parsed_config[section_name] = section_value
                    return CN.from_dict(parsed_config)
                else:
                    # Direct dictionary format
                    return CN.from_dict(config_data)
    
    # Fallback: look for config files in parent directories
    search_dirs = [checkpoint_dir, os.path.dirname(checkpoint_dir)]
    config_names = ['config.json', 'training_config.json']
    
    for search_dir in search_dirs:
        for config_name in config_names:
            config_path = os.path.join(search_dir, config_name)
            if os.path.exists(config_path):
                config_file = config_path
                break
        if config_file:
            break
    
    if config_file:
        with open(config_file, 'r') as f:
            config_dict = json.load(f)
        return CN.from_dict(config_dict)
    else:
        print(f"Warning: No configuration file found in {checkpoint_dir} or parent directories")
        print("Using default configuration - this may not match the trained model!")
        return get_default_generation_config()


def get_default_generation_config():
    """Get default configuration for generation when no config is found"""
    from train_deepspeed import get_default_config
    return get_default_config()


def load_pytorch_checkpoint(model, checkpoint_path):
    """Load PyTorch checkpoint from DeepSpeed trained model"""
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    # Try different checkpoint file patterns
    checkpoint_files = [
        os.path.join(checkpoint_path, 'mp_rank_00_model_states.pt'),
        os.path.join(checkpoint_path, 'pytorch_model.bin'),
        os.path.join(checkpoint_path, 'model.pt'),
        checkpoint_path if checkpoint_path.endswith('.pt') or checkpoint_path.endswith('.pth') else None
    ]
    
    checkpoint_file = None
    for file_path in checkpoint_files:
        if file_path and os.path.exists(file_path):
            checkpoint_file = file_path
            break
    
    if not checkpoint_file:
        raise ValueError(f"No valid checkpoint file found in or around: {checkpoint_path}")
    
    print(f"Loading checkpoint from: {checkpoint_file}")
    
    # Load state dict
    device = next(model.parameters()).device
    checkpoint = torch.load(checkpoint_file, map_location=device)
    
    # The checkpoint might have a 'module' wrapper from DeepSpeed
    if isinstance(checkpoint, dict):
        if 'module' in checkpoint:
            state_dict = checkpoint['module']
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    
    # Remove any DeepSpeed-specific prefixes if they exist
    new_state_dict = {}
    for key, value in state_dict.items():
        # Remove 'module.' prefix if it exists
        new_key = key.replace('module.', '') if key.startswith('module.') else key
        new_state_dict[new_key] = value
    
    # Load into model
    missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
    
    if missing_keys:
        print(f"Warning: Missing keys in checkpoint: {missing_keys}")
    if unexpected_keys:
        print(f"Warning: Unexpected keys in checkpoint: {unexpected_keys}")
    
    # Move model to GPU if available
    if torch.cuda.is_available():
        model = model.cuda()
        device = next(model.parameters()).device
        print(f"Model moved to device: {device}")
    
    print(f"Successfully loaded checkpoint using PyTorch")
    return model


def prepare_input_from_tokens(token_ids, max_length=None):
    """Prepare input from token ID list"""
    if not token_ids:
        raise ValueError("Empty token ID list provided")
    
    # Convert to tensor
    input_ids = torch.tensor([token_ids], dtype=torch.long)
    
    # Truncate if necessary
    if max_length and input_ids.size(1) > max_length:
        input_ids = input_ids[:, -max_length:]
    
    return input_ids


def load_prompt_tokens(prompt_tokens_str=None, prompt_file=None):
    """Load prompt token IDs from string or file"""
    if prompt_tokens_str:
        # Parse comma-separated token IDs
        try:
            token_ids = [int(x.strip()) for x in prompt_tokens_str.split(',') if x.strip()]
            return token_ids
        except ValueError as e:
            raise ValueError(f"Invalid token ID format in prompt_tokens: {e}")
    
    elif prompt_file:
        # Load from file
        try:
            with open(prompt_file, 'r') as f:
                content = f.read().strip()
            
            # Try comma-separated first
            if ',' in content:
                token_ids = [int(x.strip()) for x in content.split(',') if x.strip()]
            else:
                # Try line-by-line
                lines = content.split('\n')
                token_ids = [int(line.strip()) for line in lines if line.strip()]
            
            return token_ids
        except (FileNotFoundError, ValueError) as e:
            raise ValueError(f"Error loading prompt from file {prompt_file}: {e}")
    
    return None


def generate_with_model(model, input_ids, generation_config):
    """Generate sequences using the model"""
    # Move input to the correct device
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    
    # Set model to eval mode
    model.eval()
    
    with torch.no_grad():
        # Generate sequences using model's generate method
        generated = model.generate(
            idx=input_ids,
            max_new_tokens=generation_config.get('max_new_tokens', 50),
            temperature=generation_config.get('temperature', 1.0),
            do_sample=generation_config.get('do_sample', True),
            top_k=generation_config.get('top_k', None),
            end_token=generation_config.get('eos_token_id', None)
        )
    
    return generated


def main():
    """Main generation function"""
    parser = argparse.ArgumentParser(description='Generate sequences using PyTorch GraspGPT model (no DeepSpeed)')
    parser.add_argument('--checkpoint_path', type=str, required=True,
                       help='Path to checkpoint directory or file (e.g., /path/to/checkpoints/iter_10000)')
    parser.add_argument('--prompt_tokens', type=str, default=None,
                       help='Input prompt as comma-separated token IDs (e.g., "1,2,3,4")')
    parser.add_argument('--prompt_file', type=str, default=None,
                       help='Path to file containing prompt token IDs (one per line or comma-separated)')
    parser.add_argument('--max_new_tokens', type=int, default=50,
                       help='Maximum number of new tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.3,
                       help='Sampling temperature (1.0 = no change, >1.0 = more random, <1.0 = more deterministic)')
    parser.add_argument('--do_sample', action='store_true',
                       help='Use sampling instead of greedy decoding')
    parser.add_argument('--top_k', type=int, default=None,
                       help='Top-k sampling (only consider top k tokens)')
    parser.add_argument('--num_sequences', type=int, default=1,
                       help='Number of sequences to generate')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducible generation')
    parser.add_argument('--output_file', type=str, default=None,
                       help='Output file to save generated sequences')
    parser.add_argument('--data_path', type=str, default=None,
                       help='Path to training data (for vocabulary, overrides config)')
    
    args = parser.parse_args()
    
    print("Starting GraspGPT generation (PyTorch mode)")
    print(f"Checkpoint: {args.checkpoint_path}")
    if args.prompt_tokens:
        print(f"Prompt tokens: {args.prompt_tokens}")
    elif args.prompt_file:
        print(f"Prompt file: {args.prompt_file}")
    else:
        print("Mode: Unconditional generation")
    
    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    try:
        # Load configuration from checkpoint
        config = load_training_config(args.checkpoint_path)
        
        # Override data path if specified
        if args.data_path:
            config.dataset.data_path = args.data_path
        
        print("Configuration loaded successfully")
        print(f"Model type: {getattr(config.model, 'model_type', 'custom')}")
        print(f"Vocab size: {config.model.vocab_size}")
        print(f"Block size: {config.model.block_size}")
        
        # Debug: Print model config
        print(f"Model config debug:")
        print(f"  model_type: {getattr(config.model, 'model_type', None)}")
        print(f"  n_layer: {getattr(config.model, 'n_layer', None)}")
        print(f"  n_head: {getattr(config.model, 'n_head', None)}")
        print(f"  n_embd: {getattr(config.model, 'n_embd', None)}")
        
        # Fix model configuration - ensure proper XOR condition
        if hasattr(config.model, 'model_type') and config.model.model_type:
            # If model_type is specified, remove specific parameters to satisfy XOR
            if hasattr(config.model, 'n_layer'):
                config.model.n_layer = None
            if hasattr(config.model, 'n_head'):
                config.model.n_head = None  
            if hasattr(config.model, 'n_embd'):
                config.model.n_embd = None
        
        # Get vocabulary size from token_manager
        print("Getting vocabulary from token_manager...")
        
        try:
            # Import token_manager
            from model.token_manager import get_token_manager
            
            # Get token manager
            token_manager = get_token_manager()
            
            # Generate mapping based on volume dimensions (from config or reasonable defaults)
            # Default volume dimensions if not in config
            img_h, img_w, img_d = 54, 36, 20  # Default dimensions
            
            # Try to get actual dimensions from a sample data file if available
            if hasattr(config.dataset, 'data_path') and config.dataset.data_path:
                try:
                    import glob
                    
                    data_files = glob.glob(os.path.join(config.dataset.data_path, "*.pth"))
                    if data_files:
                        # Load first file to get dimensions
                        sample_file = data_files[0]
                        print(f"Loading sample file to get volume dimensions: {os.path.basename(sample_file)}")
                        raw_data = torch.load(sample_file, weights_only=False)
                        if 'volume_dims' in raw_data:
                            img_h, img_w, img_d = raw_data['volume_dims']
                            print(f"Got volume dimensions from data: {img_h}x{img_w}x{img_d}")
                except Exception as e:
                    print(f"Could not load dimensions from data, using defaults: {e}")
            
            # Generate token mapping
            token_mapping = token_manager.generate_mapping(img_h, img_w, img_d)
            vocab_size = len(token_mapping)
            
            # Update config
            config.model.vocab_size = vocab_size
            
            print(f"Vocabulary size from token_manager: {vocab_size}")
            print(f"Volume dimensions used: {img_h}x{img_w}x{img_d}")
            print("Skipped loading full dataset - using token_manager for vocab only")
            
        except Exception as e:
            # If token_manager approach fails, we need vocab_size from config or fail
            print(f"Token manager approach failed ({e})")
            print("Error: Cannot determine vocabulary size. Please ensure config has vocab_size.")
            raise ValueError(f"Failed to get vocabulary size: {e}")
        
        # Create model
        print("Creating model...")
        model = graspGPT(config.model)
        print(f"Model created: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters")
        
        # Load checkpoint
        model = load_pytorch_checkpoint(model, args.checkpoint_path)
        
        # Prepare generation configuration
        generation_config = {
            'max_new_tokens': args.max_new_tokens,
            'temperature': args.temperature,
            'do_sample': args.do_sample,
            'top_k': args.top_k,
            'eos_token_id': token_mapping['end'] if 'token_mapping' in locals() else None
        }
        
        # Load prompt tokens
        prompt_token_ids = load_prompt_tokens(args.prompt_tokens, args.prompt_file)
        
        # Generate sequences
        print(f"Starting generation of {args.num_sequences} sequences...")
        if prompt_token_ids:
            print(f"Using prompt tokens: {prompt_token_ids}")
        else:
            print("Using unconditional generation (no prompt)")
        
        all_outputs = []
        
        for i in range(args.num_sequences):
            print(f"Generating sequence {i+1}/{args.num_sequences}...")
            
            # Prepare input
            if prompt_token_ids:
                input_ids = prepare_input_from_tokens(prompt_token_ids, config.model.block_size)
            else:
                # Unconditional generation - start with a special start token
                start_token_id = token_mapping.get('object32', 1)  # Use object09 as start or fallback to token 1
                input_ids = torch.tensor([[start_token_id]], dtype=torch.long)
            
            original_length = input_ids.size(1)
            
            if not prompt_token_ids:  # Only print for unconditional generation
                print(f"Input Prompt: [{start_token_id}]")
            
            # Generate
            start_time = time.time()
            generated = generate_with_model(model, input_ids, generation_config)
            generation_time = time.time() - start_time
            
            print(f"Generation {i+1} completed in {generation_time:.2f}s")
            print(f"Generated sequence length: {generated.size(1)} tokens")
            
            # Decode output - just return the new token IDs
            new_tokens = generated[:, original_length:]
            if len(new_tokens.shape) == 3:  # Handle 3D tensor case
                decoded = new_tokens[0, :, 0].cpu().numpy().tolist()
            else:  # Handle 2D tensor case
                decoded = new_tokens[0].cpu().numpy().tolist()
            
            all_outputs.append({
                'sequence_id': i + 1,
                'prompt_tokens': prompt_token_ids,
                'generated_tokens': decoded,
                'total_length': generated.size(1),
                'new_tokens': generated.size(1) - original_length,
                'generation_time': generation_time
            })
            
            # Try to decode and save as 3D object
            try:
                from model.token_manager import decode_sequence
                from model.core import save_voxels
                full_sequence = input_ids[0].cpu().numpy().tolist() + decoded
                decoded_obj = decode_sequence(full_sequence, token_mapping)
                save_voxels(decoded_obj, f'pred_{i+1}.obj')
                print(f"Generated output saved as pred_{i+1}.obj: {decoded_obj}")
            except Exception as e:
                print(f"Could not decode/save 3D object: {e}")
                print(f"Generated tokens: {decoded}")
            
            print("-" * 50)
        
        # Save outputs to file if specified
        if args.output_file:
            output_data = {
                'generation_config': generation_config,
                'model_config': {
                    'model_type': getattr(config.model, 'model_type', 'custom'),
                    'vocab_size': config.model.vocab_size,
                    'block_size': config.model.block_size,
                    'n_layer': getattr(config.model, 'n_layer', None),
                    'n_head': getattr(config.model, 'n_head', None),
                    'n_embd': getattr(config.model, 'n_embd', None)
                },
                'checkpoint_path': args.checkpoint_path,
                'outputs': all_outputs
            }
            
            with open(args.output_file, 'w') as f:
                json.dump(output_data, f, indent=2)
            
            print(f"Results saved to: {args.output_file}")
        
        print("Generation completed successfully!")
    
    except KeyboardInterrupt:
        print("\nGeneration interrupted by user")
    except Exception as e:
        print(f"Generation failed with error: {str(e)}")
        raise


if __name__ == "__main__":
    main()