#!/usr/bin/env python3
"""
DeepSpeed model generation script for GraspGPT

This script loads a model trained with DeepSpeed and generates sequences.
It supports both single-GPU and multi-GPU generation with proper checkpoint loading.

使用方法：

  1. 通过命令行直接提供 token IDs（逗号分隔）：

  python graspGPT/generate_deepspeed.py \
    --checkpoint_path /path/to/checkpoints/iter_10000 \
    --prompt_tokens "1,2,3,4,5" \
    --max_new_tokens 50 \
    --do_sample

  2. 通过文件提供 token IDs（逗号分隔或每行一个）：

  python graspGPT/generate_deepspeed.py \
    --checkpoint_path /path/to/checkpoints/iter_10000 \
    --prompt_file example_prompt.txt \
    --temperature 0.8

  3. 无条件生成（不提供 prompt）：

  python graspGPT/generate_deepspeed.py \
    --checkpoint_path /path/to/checkpoints/iter_10000 \
    --num_sequences 3

  4. 多GPU分布式生成：

  deepspeed graspGPT/generate_deepspeed.py \
    --checkpoint_path /path/to/checkpoints/iter_10000 \
    --prompt_tokens "10,20,30,40" \
    --num_sequences 5 \
    --output_file results.json

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
import deepspeed
from deepspeed import comm as dist

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


def load_deepspeed_checkpoint(model, checkpoint_path, deepspeed_config, use_deepspeed=True):
    """Load DeepSpeed checkpoint with optional fallback to regular PyTorch loading"""
    rank = dist.get_rank() if dist.is_initialized() else 0
    
    if rank == 0:
        print(f"Loading checkpoint from: {checkpoint_path}")
    
    # Parse checkpoint path
    if os.path.isfile(checkpoint_path):
        # Handle case where full path to a specific file is given
        parent_dir = os.path.dirname(checkpoint_path)
        tag = os.path.basename(checkpoint_path)
    elif os.path.isdir(checkpoint_path):
        # Handle case where directory is given
        parent_dir = os.path.dirname(checkpoint_path)
        tag = os.path.basename(checkpoint_path)
    else:
        raise ValueError(f"Checkpoint path {checkpoint_path} does not exist")
    
    if use_deepspeed:
        try:
            # Initialize DeepSpeed engine for inference
            model_engine, _, _, _ = deepspeed.initialize(
                model=model,
                config=deepspeed_config,
                model_parameters=model.parameters()
            )
            
            # Load the checkpoint
            _, client_state = model_engine.load_checkpoint(parent_dir, tag=tag)
            if rank == 0:
                print(f"Successfully loaded checkpoint from {checkpoint_path} using DeepSpeed")
            
            return model_engine
            
        except Exception as e:
            if rank == 0:
                print(f"DeepSpeed loading failed: {e}")
                print("Falling back to regular PyTorch loading...")
            use_deepspeed = False
    
    if not use_deepspeed:
        # Fallback: Load using regular PyTorch
        checkpoint_file = os.path.join(checkpoint_path, 'mp_rank_00_model_states.pt')
        if os.path.exists(checkpoint_file):
            if rank == 0:
                print(f"Loading checkpoint using PyTorch from: {checkpoint_file}")
            
            # Load state dict and move to appropriate device
            device = next(model.parameters()).device
            checkpoint = torch.load(checkpoint_file, map_location=device)
            
            # The checkpoint might have a 'module' wrapper
            state_dict = checkpoint.get('module', checkpoint)
            
            # Load into model
            model.load_state_dict(state_dict, strict=False)
            
            # Ensure model is on correct device
            if torch.cuda.is_available():
                model = model.cuda()
                device = next(model.parameters()).device
                if rank == 0:
                    print(f"Model moved to device: {device}")
            
            if rank == 0:
                print(f"Successfully loaded checkpoint using PyTorch")
            
            # Return model wrapped as if it were a DeepSpeed engine
            class ModelWrapper:
                def __init__(self, model):
                    self.module = model
                    self.model = model
                    self.local_rank = 0
                    
                def eval(self):
                    self.module.eval()
                    
                def __call__(self, *args, **kwargs):
                    return self.module(*args, **kwargs)
            
            return ModelWrapper(model)
        else:
            raise ValueError(f"Checkpoint file not found: {checkpoint_file}")
    
    return model_engine


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


def generate_with_model(model_engine, input_ids, generation_config):
    """Generate sequences using the model"""
    rank = dist.get_rank() if dist.is_initialized() else 0
    
    # Move input to the correct device
    if hasattr(model_engine, 'local_rank') and model_engine.local_rank is not None:
        device = f'cuda:{model_engine.local_rank}'
    else:
        # Fallback: get device from model parameters
        device = next(model_engine.module.parameters()).device
    
    input_ids = input_ids.to(device)
    
    # Set model to eval mode
    model_engine.eval()
    
    with torch.no_grad():
        # Generate sequences
        if hasattr(model_engine.module, 'generate'):
            # Use model's generate method if available
            # Use graspGPT.generate() method - now fixed to work with transformers properly
            generated = model_engine.module.generate(
                idx=input_ids,
                max_new_tokens=generation_config.get('max_new_tokens', 50),
                temperature=generation_config.get('temperature', 1.0),
                do_sample=generation_config.get('do_sample', True),
                top_k=generation_config.get('top_k', None),
                end_token=generation_config.get('eos_token_id', None)
            )
        else:
            # If model doesn't have generate method, this shouldn't happen with graspGPT
            raise ValueError("Model does not have a generate method")
    
    return generated




def main():
    """Main generation function"""
    parser = argparse.ArgumentParser(description='Generate sequences using DeepSpeed-trained GraspGPT model')
    parser.add_argument('--checkpoint_path', type=str, required=True,
                       help='Path to checkpoint directory (e.g., /path/to/checkpoints/iter_10000)')
    parser.add_argument('--prompt_tokens', type=str, default=None,
                       help='Input prompt as comma-separated token IDs (e.g., "1,2,3,4")')
    parser.add_argument('--prompt_file', type=str, default=None,
                       help='Path to file containing prompt token IDs (one per line or comma-separated)')
    parser.add_argument('--max_new_tokens', type=int, default=50,
                       help='Maximum number of new tokens to generate')
    parser.add_argument('--temperature', type=float, default=1.0,
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
    parser.add_argument('--deepspeed_config', type=str, default=None,
                       help='Path to DeepSpeed config file (overrides default)')
    parser.add_argument('--local_rank', type=int, default=-1,
                       help='Local rank for distributed generation')
    
    # Parse arguments
    args, unknown_args = parser.parse_known_args()
    
    # Initialize DeepSpeed if running in distributed mode
    try:
        deepspeed.init_distributed()
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        is_distributed = True
    except:
        rank = 0
        world_size = 1
        is_distributed = False
    
    if rank == 0:
        print(f"Starting GraspGPT generation (rank {rank}, world_size {world_size})")
        print(f"Checkpoint: {args.checkpoint_path}")
        if args.prompt_tokens:
            print(f"Prompt tokens: {args.prompt_tokens}")
        elif args.prompt_file:
            print(f"Prompt file: {args.prompt_file}")
        else:
            print("Mode: Unconditional generation")
    
    # Set random seed
    torch.manual_seed(args.seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed + rank)
    
    try:
        # Load configuration from checkpoint
        config = load_training_config(args.checkpoint_path)
        
        # Override data path if specified
        if args.data_path:
            config.dataset.data_path = args.data_path
        
        if rank == 0:
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
        
        # Get vocabulary size from token_manager (much faster than loading dataset)
        if rank == 0:
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
                        if rank == 0:
                            print(f"Loading sample file to get volume dimensions: {os.path.basename(sample_file)}")
                        raw_data = torch.load(sample_file, weights_only=False)
                        if 'volume_dims' in raw_data:
                            img_h, img_w, img_d = raw_data['volume_dims']
                            if rank == 0:
                                print(f"Got volume dimensions from data: {img_h}x{img_w}x{img_d}")
                except Exception as e:
                    if rank == 0:
                        print(f"Could not load dimensions from data, using defaults: {e}")
            
            # Generate token mapping
            token_mapping = token_manager.generate_mapping(img_h, img_w, img_d)
            vocab_size = len(token_mapping)
            
            # Update config
            config.model.vocab_size = vocab_size
            
            if rank == 0:
                print(f"Vocabulary size from token_manager: {vocab_size}")
                print(f"Volume dimensions used: {img_h}x{img_w}x{img_d}")
                print("Skipped loading full dataset - using token_manager for vocab only")
            
            dataset = None  # No need to keep dataset object
            
        except Exception as e:
            # If token_manager approach fails, we need vocab_size from config or fail
            if rank == 0:
                print(f"Token manager approach failed ({e})")
                print("Error: Cannot determine vocabulary size. Please ensure config has vocab_size.")
            raise ValueError(f"Failed to get vocabulary size: {e}")
            
            dataset = None
        
        # Create model
        if rank == 0:
            print("Creating model...")
        
        model = graspGPT(config.model)
        
        if rank == 0:
            print(f"Model created: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters")
        
        # Create DeepSpeed configuration for inference
        deepspeed_config_path = args.deepspeed_config or config.deepspeed.config_path
        with open(deepspeed_config_path, 'r') as f:
            ds_config = json.load(f)
        
        # Configure for inference
        ds_config.update({
            "train_batch_size": 1,
            "train_micro_batch_size_per_gpu": 1,
            "gradient_accumulation_steps": 1,
        })
        
        # Load checkpoint
        model_engine = load_deepspeed_checkpoint(model, args.checkpoint_path, ds_config)
        
        # Prepare generation configuration
        generation_config = {
            'max_new_tokens': args.max_new_tokens,
            'temperature': args.temperature,
            'do_sample': args.do_sample,
            'top_k': args.top_k,
            'eos_token_id': token_mapping['end'] if 'token_mapping' in locals() else None  # Use eos_token_id instead of end_token
        }
        
        # Load prompt tokens
        prompt_token_ids = load_prompt_tokens(args.prompt_tokens, args.prompt_file)
        
        # Generate sequences
        if rank == 0:
            print(f"Starting generation of {args.num_sequences} sequences...")
            if prompt_token_ids:
                print(f"Using prompt tokens: {prompt_token_ids}")
            else:
                print("Using unconditional generation (no prompt)")
        
        all_outputs = []
        
        for i in range(args.num_sequences):
            if rank == 0:
                print(f"Generating sequence {i+1}/{args.num_sequences}...")
            
            # Prepare input
            if prompt_token_ids:
                input_ids = prepare_input_from_tokens(prompt_token_ids, config.model.block_size)
            else:
                # Unconditional generation - start with a special start token or random token
                start_token_id = token_mapping['object09'] 
                input_ids = torch.tensor([[start_token_id]], dtype=torch.long)  # Use last token as start
            
            original_length = input_ids.size(1)

            print(f"Input Prompt: [{start_token_id}]")
            # Generate
            start_time = time.time()
            generated = generate_with_model(model_engine, input_ids, generation_config)
            generation_time = time.time() - start_time
            
            if rank == 0:
                print(f"Generation {i+1} completed in {generation_time:.2f}s")
                print(f"Generated sequence length: {generated.size(1)} tokens")
                
                # Decode output - just return the new token IDs
                new_tokens = generated[:, original_length:]
                decoded = new_tokens[0,:,0].cpu().numpy().tolist()
                all_outputs.append({
                    'sequence_id': i + 1,
                    'prompt_tokens': prompt_token_ids,
                    'generated_tokens': decoded,
                    'total_length': generated.size(1),
                    'new_tokens': generated.size(1) - original_length,
                    'generation_time': generation_time
                })
                
                from model.token_manager import  decode_sequence
                from model.core import save_voxels
                decoded = decode_sequence(input_ids[0].cpu().numpy().tolist() +decoded, token_mapping)
                save_voxels( decoded, 'pred.obj')
                print(f"Generated output: {decoded}")
                print("-" * 50)
        
        # Save outputs to file if specified
        if args.output_file and rank == 0:
            output_data = {
                'generation_config': generation_config,
                'model_config': {
                    'model_type': getattr(config.model, 'model_type', 'custom'),
                    'vocab_size': config.model.vocab_size,
                    'block_size': config.model.block_size,
                    'n_layer': config.model.n_layer,
                    'n_head': config.model.n_head,
                    'n_embd': config.model.n_embd
                },
                'checkpoint_path': args.checkpoint_path,
                'outputs': all_outputs
            }
            
            with open(args.output_file, 'w') as f:
                json.dump(output_data, f, indent=2)
            
            print(f"Results saved to: {args.output_file}")
        
        if rank == 0:
            print("Generation completed successfully!")
    
    except KeyboardInterrupt:
        if rank == 0:
            print("\nGeneration interrupted by user")
    except Exception as e:
        if rank == 0:
            print(f"Generation failed with error: {str(e)}")
        raise
    finally:
        # Cleanup distributed resources
        if is_distributed and dist.is_initialized():
            try:
                dist.destroy_process_group()
                if rank == 0:
                    print("Distributed process group cleaned up")
            except Exception as e:
                if rank == 0:
                    print(f"Warning: Failed to cleanup process group: {e}")


if __name__ == "__main__":
    main()