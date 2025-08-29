"""
Full definition of a GPT Language Model, all of it in this single file.

References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math

import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoConfig, Qwen2ForCausalLM, Qwen2Config

from .utils import CfgNode as CN


# -----------------------------------------------------------------------------



class graspGPT(nn.Module):
    """
    GraspGPT: A flexible GPT-based language model with support for RoPE (Rotary Position Embedding)
    
    This model supports both traditional absolute position encoding and RoPE (Rotary Position Embedding).
    The model architecture is based on the GPT-2 transformer with the following key features:
    
    - Configurable model size (layers, heads, embedding dimensions)
    - Support for predefined model types (gpt2, gpt-mini, etc.) or custom parameters
    - RoPE integration for better long-sequence modeling
    - Flash attention support for improved efficiency
    - Flexible configuration system using CfgNode
    
    Configuration Options:
    - model_type: Predefined model size ('gpt2', 'gpt-mini', etc.) OR None to use custom params
    - n_layer, n_head, n_embd: Custom model architecture parameters
    - vocab_size: Vocabulary size for the tokenizer
    - block_size: Maximum sequence length
    - use_rope: Whether to use Rotary Position Embedding (True) or absolute position encoding (False)
    - dropout parameters: embd_pdrop, resid_pdrop, attn_pdrop for regularization
    
    Note: Either model_type OR (n_layer, n_head, n_embd) must be specified, but not both.
    """

    @staticmethod
    def get_default_config():
        """
        Get the default configuration for GraspGPT model.
        
        Returns:
            CN: Configuration node with default settings including:
                - model_type: Default model size ('gpt-mini')
                - Architecture params: n_layer, n_head, n_embd (set to None for custom config)
                - Required params: vocab_size, block_size (must be set externally)
                - Dropout rates: embd_pdrop, resid_pdrop, attn_pdrop (0.1 each)
                - RoPE settings: use_rope (True), rope_base (10000.0)
                - Flash attention: use_flash_attention (True)
        """
        C = CN()
        # either model_type or (n_layer, n_head, n_embd) must be given in the config
        C.model_type = 'gpt-mini'
        C.n_layer = None
        C.n_head = None
        C.n_embd =  None
        # these options must be filled in externally
        C.vocab_size = None
        C.block_size = None
        # dropout hyperparameters
        C.embd_pdrop = 0.1
        C.resid_pdrop = 0.1
        C.attn_pdrop = 0.1
        # transformers model name
        C.transformer_model_name = 'gpt2'
        # flash attention settings
        C.use_flash_attention = True
        # RoPE settings
        C.use_rope = True
        C.rope_base = 10000.0
        return C

    def __init__(self, config):
        """
        Initialize the GraspGPT model.
        
        Args:
            config (CN): Configuration object containing model parameters.
                        Must have vocab_size and block_size set.
                        Must specify either model_type OR (n_layer, n_head, n_embd).
        
        Raises:
            AssertionError: If required config parameters are missing or if both
                           model_type and custom parameters are provided.
        """
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.block_size = config.block_size
        self.config = config

        # Ensure exactly one configuration method is used (XOR)
        type_given = config.model_type is not None
        params_given = all([config.n_layer is not None, config.n_head is not None, config.n_embd is not None])
        assert type_given ^ params_given # exactly one of these (XOR)
        if type_given:
            # translate from model_type to detailed configuration
            config.merge_from_dict({
                # names follow the huggingface naming conventions
                # GPT-1
                'openai-gpt':   dict(n_layer=12, n_head=12, n_embd=768),  # 117M params
                # GPT-2 configs
                'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
                'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
                'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
                'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
                # Gophers
                'gopher-44m':   dict(n_layer=8, n_head=16, n_embd=512),
                # (there are a number more...)
                # I made these tiny models up
                'gpt-my_mini':     dict(n_layer=6, n_head=8, n_embd=512),
                'gpt-mini':     dict(n_layer=6, n_head=8, n_embd=256),
                'gpt-micro':    dict(n_layer=4, n_head=4, n_embd=128),
                'gpt-nano':     dict(n_layer=3, n_head=3, n_embd=48),
            }[config.model_type])

        # Check if we should use RoPE
        use_rope = getattr(config, 'use_rope', True)
        
        if use_rope:
            # Use Qwen2 from transformers library
            print("Using Qwen2 model with RoPE position encoding")
            # Create Qwen2 config with proper head configuration
            qwen_config = Qwen2Config(
                vocab_size=config.vocab_size[0] if isinstance(config.vocab_size, list) else config.vocab_size,
                hidden_size=config.n_embd,
                intermediate_size=4 * config.n_embd,
                num_hidden_layers=config.n_layer,
                num_attention_heads=config.n_head,
                num_key_value_heads=config.n_head,  # Set key-value heads equal to attention heads
                max_position_embeddings=config.block_size,
                rms_norm_eps=1e-6,
                tie_word_embeddings=True,
                attention_dropout=config.attn_pdrop,
                # RoPE specific settings
                rope_theta=getattr(config, 'rope_base', 10000.0),
                # Flash Attention 2 configuration
                attn_implementation="flash_attention_2" if getattr(config, 'use_flash_attention', True) else "eager",
            )
            self.model = Qwen2ForCausalLM(qwen_config)
            self.num_heads = 1
        else:
            # Use traditional transformers implementation
            print("Using traditional absolute position encoding")
            # Initialize transformers model
            self.transformer_model_name = getattr(config, 'transformer_model_name', 'gpt2')
            # Create model config and initialize model
            model_config = AutoConfig.from_pretrained(self.transformer_model_name)
            # Update config with our custom parameters
            model_config.vocab_size = config.vocab_size[0] if isinstance(config.vocab_size, list) else config.vocab_size
            model_config.n_positions = config.block_size
            model_config.n_embd = config.n_embd
            model_config.n_layer = config.n_layer
            model_config.n_head = config.n_head
            model_config.embd_pdrop = config.embd_pdrop
            model_config.resid_pdrop = config.resid_pdrop
            model_config.attn_pdrop = config.attn_pdrop
            
            # Enable Flash Attention 2 if requested and available
            use_flash_attention = getattr(config, 'use_flash_attention', True)
            if use_flash_attention:
                try:
                    import flash_attn
                    # Set Flash Attention 2 configuration
                    model_config.attn_implementation = "flash_attention_2"
                    print("Flash Attention 2 enabled")
                except ImportError:
                    print("Warning: Flash Attention 2 requested but package not found. "
                          "Install with: pip install flash-attn --no-build-isolation")
            
            self.model = AutoModelForCausalLM.from_config(model_config)
            
            # Custom heads for multi-task learning
            self.num_heads = len(config.vocab_size) if isinstance(config.vocab_size, list) else 1
            if self.num_heads > 1:
               raise ValueError("Multi-head output is not supported for now")

        # report number of parameters
        n_params = sum(p.numel() for p in self.model.parameters())
        config.n_params = n_params
        print("number of parameters: %.2fM" % (n_params/1e6,))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

   

    def configure_optimizers(self, train_config):
        """
        Use transformers' standard optimizer configuration
        Compatible with both regular training and DeepSpeed
        """
        try:
            # Try to use transformers' built-in optimizer grouping
            from transformers.optimization import get_optimizer_grouped_parameters
            
            optimizer_grouped_parameters = get_optimizer_grouped_parameters(
                self.model,
                weight_decay=train_config.weight_decay,
                learning_rate=train_config.learning_rate,
            )
            
            optimizer = torch.optim.AdamW(
                optimizer_grouped_parameters,
                lr=train_config.learning_rate,
                betas=getattr(train_config, 'betas', (0.9, 0.95))
            )
            
        except ImportError:
            # Fallback to manual parameter grouping if transformers function not available
            decay_params = []
            no_decay_params = []
            
            for name, param in self.named_parameters():
                if param.requires_grad:
                    # Apply weight decay to Linear weights only
                    if 'weight' in name and any(layer in name for layer in ['linear', 'dense', 'c_fc', 'c_proj', 'c_attn']):
                        decay_params.append(param)
                    else:
                        # No weight decay for biases, LayerNorm, Embedding, etc.
                        no_decay_params.append(param)
            
            # Create optimizer groups
            optimizer_groups = [
                {"params": decay_params, "weight_decay": train_config.weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ]
            
            optimizer = torch.optim.AdamW(
                optimizer_groups,
                lr=train_config.learning_rate,
                betas=getattr(train_config, 'betas', (0.9, 0.95))
            )
        
        return optimizer
    
    def get_parameter_groups(self, train_config):
        """
        Get parameter groups for DeepSpeed optimizer configuration
        Returns parameter groups that can be used by DeepSpeed
        """
        try:
            # Try to use transformers' built-in optimizer grouping
            from transformers.optimization import get_optimizer_grouped_parameters
            
            optimizer_grouped_parameters = get_optimizer_grouped_parameters(
                self.model,
                weight_decay=train_config.weight_decay,
                learning_rate=train_config.learning_rate,
            )
            
            return optimizer_grouped_parameters
            
        except ImportError:
            # Fallback to manual parameter grouping if transformers function not available
            decay_params = []
            no_decay_params = []
            
            for name, param in self.named_parameters():
                if param.requires_grad:
                    # Apply weight decay to Linear weights only
                    if 'weight' in name and any(layer in name for layer in ['linear', 'dense', 'c_fc', 'c_proj', 'c_attn']):
                        decay_params.append(param)
                    else:
                        # No weight decay for biases, LayerNorm, Embedding, etc.
                        no_decay_params.append(param)
            
            # Create optimizer groups
            optimizer_groups = [
                {"params": decay_params, "weight_decay": train_config.weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ]
            
            return optimizer_groups

    def forward(self, idx, targets=None, attention_mask=None):
        """
        Args:
            idx: input tensor of shape (batch_size, sequence_length, num_features)
            targets: target tensor of shape (batch_size, sequence_length, num_features)
            attention_mask: mask tensor of shape (batch_size, sequence_length)
        """
        device = idx.device
        use_rope = getattr(self.config, 'use_rope', True)
        
        if use_rope:
            # RoPE version expects 2D input (batch_size, sequence_length)
            b, t, g = idx.size()
            assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"
            
            # Convert idx to token format for RoPE model
            input_ids = idx[..., 0].long()  # Use first feature as main input
            
            

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
            logits = outputs.logits
            loss = None
            
            # Create list of logits for multi-head output
            logits_heads = [logits]

            loss = None
            if targets is not None:
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), 
                                     targets[..., 0].long().reshape(-1), 
                                     ignore_index=-1)
            
            return logits_heads, loss
        else:
            # Original transformers version
            b, t, g = idx.size()
            assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"
            
            # Convert idx to token format expected by transformers
            input_ids = idx[..., 0].long()  # Use first feature as main input
            
            # Ensure attention_mask is properly formatted for Flash Attention 2
            if attention_mask is not None:
                if attention_mask.dtype == torch.long or attention_mask.dtype == torch.int:
                    attention_mask = attention_mask.to(torch.bool)
            
            # Forward through transformers model
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
            
            # Get logits from the model
            logits = outputs.logits
            
            # Create list of logits for multi-head output
            logits_heads = [logits]
            
            # Calculate loss if targets provided
            loss = None
            if targets is not None:
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), 
                                     targets[..., 0].long().reshape(-1), 
                                     ignore_index=-1)

            return logits_heads, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, do_sample=False, top_k=None, end_token=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        use_rope = getattr(self.config, 'use_rope', True)
        
        if use_rope:
            # RoPE version
            input_ids = idx[..., 0].long() if idx.dim() > 2 else idx.long()
            
            # Prepare generation config for transformers
            generation_config = {
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "do_sample": do_sample,
                "return_dict_in_generate": True,
                "output_scores": True,
                "use_cache": True
            }
            
            # Add optional parameters only if they're not None
            if top_k is not None:
                generation_config["top_k"] = top_k
            if end_token is not None:
                generation_config["eos_token_id"] = end_token
                generation_config["pad_token_id"] = 0
            
            # Use transformers generate method
            generated = self.model.generate(
                input_ids=input_ids,
                **generation_config
            )
            
            generated_ids = generated.sequences
        else:
            # Original transformers version
            input_ids = idx[..., 0].long() if idx.dim() > 2 else idx.long()
            
            generation_config = {
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "do_sample": do_sample,
                "top_k": top_k,
                "pad_token_id": end_token,
                "eos_token_id": end_token,
                "return_dict_in_generate": True,
                "output_scores": True,
                "use_cache": True
            }
            
            generated = self.model.generate(
                input_ids=input_ids,
                **generation_config
            )
            
            generated_ids = generated.sequences
        
        # Always return in the expected format: (batch_size, sequence_length, num_features)
        # Add the feature dimension to match the original input format
        if idx.dim() > 2:
            # If input has multiple features, expand to match
            expanded = generated_ids.unsqueeze(-1).expand(-1, -1, idx.size(-1))
            return expanded
        else:
            # If input is 2D, add a single feature dimension
            return generated_ids.unsqueeze(-1)

    def save_model(self, path):
        """Save model state dict to file"""
        # Save the transformers model
        state_dict = {
            'model_state_dict': self.model.state_dict(),
            'config': self.config
        }
        torch.save(state_dict, path)

    def load_model(self, path):
        """Load model state dict from file"""
        # Load to current device
        device = next(self.parameters()).device
        state_dict = torch.load(path, map_location=device)
        
        if 'model_state_dict' in state_dict:
            # Load transformers model
            self.model.load_state_dict(state_dict['model_state_dict'])
        else:
            # Fallback for old format
            self.load_state_dict(state_dict)
        
        self.eval()
        
    def is_using_flash_attention(self):
        """Check if the model is using Flash Attention 2"""
        try:
            # Check the model config
            if hasattr(self.model.config, 'attn_implementation'):
                if self.model.config.attn_implementation == 'flash_attention_2':
                    # Verify flash_attn is actually installed
                    try:
                        import flash_attn
                        return True
                    except ImportError:
                        return False
            return False
        except Exception:
            return False