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


class RoPEPositionEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) implementation.
    
    RoPE encodes positional information by rotating the query and key vectors
    in the attention mechanism. This allows for better extrapolation to longer
    sequences than traditional absolute position embeddings.
    
    References:
    - RoFormer: Enhanced Transformer with Rotary Position Embedding
    - https://arxiv.org/abs/2104.09864
    
    Args:
        dim (int): Dimension of the embedding (typically head_dim)
        max_position_embeddings (int): Maximum sequence length to precompute embeddings for
        base (float): Base for the frequency calculation (default: 10000.0)
    """
    
    def __init__(self, dim, max_position_embeddings=2048, base=10000.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
        # Precompute the frequency for each dimension
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq)
        
        # Cache for position embeddings
        self._cos_cached = None
        self._sin_cached = None
        self._seq_len_cached = 0
    
    def _get_cos_sin(self, seq_len, device, dtype):
        """Generate cosine and sine values for RoPE"""
        if (self._cos_cached is None or 
            self._seq_len_cached < seq_len or 
            self._cos_cached.device != device or
            self._cos_cached.dtype != dtype):
            
            self._seq_len_cached = max(seq_len, self._seq_len_cached)
            
            # Generate position indices
            t = torch.arange(self._seq_len_cached, device=device, dtype=self.inv_freq.dtype)
            
            # Compute frequencies
            freqs = torch.einsum("i,j->ij", t, self.inv_freq.to(device))
            
            # Create the rotation matrix components
            emb = torch.cat((freqs, freqs), dim=-1)
            self._cos_cached = emb.cos().to(dtype)
            self._sin_cached = emb.sin().to(dtype)
        
        return self._cos_cached[:seq_len], self._sin_cached[:seq_len]
    
    def _rotate_half(self, x):
        """Rotate half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
    
    def apply_rotary_pos_emb(self, q, k, position_ids=None):
        """
        Apply rotary position embedding to query and key tensors.
        
        Args:
            q (torch.Tensor): Query tensor of shape [batch, heads, seq_len, head_dim]
            k (torch.Tensor): Key tensor of shape [batch, heads, seq_len, head_dim]
            position_ids (torch.Tensor, optional): Position indices (unused in current implementation)
            
        Returns:
            tuple: Rotated (query, key) tensors with position information encoded
        """
        seq_len = q.shape[-2]
        cos, sin = self._get_cos_sin(seq_len, q.device, q.dtype)
        
        if position_ids is not None:
            cos = cos[position_ids]
            sin = sin[position_ids]
        else:
            cos = cos[:seq_len]
            sin = sin[:seq_len]
        
        # Ensure cos and sin have the right shape for broadcasting
        cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, dim]
        sin = sin.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, dim]
        
        q_embed = (q * cos) + (self._rotate_half(q) * sin)
        k_embed = (k * cos) + (self._rotate_half(k) * sin)
        
        return q_embed, k_embed


class RoPEAttention(nn.Module):
    """
    Multi-head attention mechanism with Rotary Position Embedding (RoPE).
    
    This module implements the core attention mechanism where position information
    is encoded through RoPE rotations rather than additive position embeddings.
    The attention weights are computed as usual, but query and key vectors are
    rotated based on their positions before computing attention scores.
    
    Args:
        config: Configuration object containing:
            - n_head: Number of attention heads
            - n_embd: Embedding dimension
            - attn_pdrop: Attention dropout probability
            - rope_base: Base frequency for RoPE (default: 10000.0)
            - block_size: Maximum sequence length
    """
    
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.attn_pdrop
        
        assert self.n_embd % self.n_head == 0
        self.head_dim = self.n_embd // self.n_head
        
        # Query, key, value projections for all heads
        self.c_attn = nn.Linear(self.n_embd, 3 * self.n_embd, bias=True)
        # Output projection
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=True)
        # Regularization
        self.attn_dropout = nn.Dropout(self.dropout)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        
        # RoPE position embedding
        self.rope = RoPEPositionEmbedding(
            dim=self.head_dim,
            max_position_embeddings=config.block_size
        )
        
        # Flash attention flag
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        
        # Causal mask
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))
    
    def forward(self, x, attention_mask=None):
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality
        
        # Calculate query, key, values for all heads in batch
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        
        # Reshape for multi-head attention
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)
        
        # Apply RoPE to query and key
        q, k = self.rope.apply_rotary_pos_emb(q, k)
        
        # Causal self-attention
        if self.flash:
            # Use Flash Attention if available
            # Convert attention_mask from [B, T] to [B, 1, T, T] for scaled_dot_product_attention
            attn_mask_4d = None
            if attention_mask is not None:
                # Handle different input dimensions of attention_mask
                if attention_mask.dim() == 2:  # [B, T]
                    attn_mask_4d = attention_mask.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, T]
                    attn_mask_4d = attn_mask_4d.expand(-1, -1, T, -1)  # [B, 1, T, T]
                elif attention_mask.dim() == 4:  # Already [B, 1, T, T] or similar
                    attn_mask_4d = attention_mask
                else:
                    # Squeeze extra dimensions and reshape to [B, T] then convert to 4D
                    attn_mask_4d = attention_mask.squeeze()
                    if attn_mask_4d.dim() == 2:  # Now [B, T]
                        attn_mask_4d = attn_mask_4d.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, T]
                        attn_mask_4d = attn_mask_4d.expand(-1, -1, T, -1)  # [B, 1, T, T]
                    else:
                        raise ValueError(f"Unsupported attention_mask dimensions: {attention_mask.shape}")
                # Convert boolean mask to float mask for attention
                attn_mask_4d = attn_mask_4d.float().masked_fill(~attn_mask_4d.bool(), float('-inf'))
            
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, 
                attn_mask=attn_mask_4d, 
                dropout_p=self.dropout if self.training else 0,
                is_causal=True
            )
        else:
            # Manual implementation
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            if attention_mask is not None:
                att = att + attention_mask
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side
        
        # Output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class RoPETransformerBlock(nn.Module):
    """
    Transformer block with RoPE attention and MLP.
    
    This is a standard transformer block that uses RoPE attention instead of
    traditional attention with additive position embeddings. The architecture
    follows the GPT-2 design with pre-layer normalization.
    
    Architecture:
    1. Layer norm -> RoPE Attention -> Residual connection
    2. Layer norm -> MLP -> Residual connection
    
    Args:
        config: Configuration object containing model parameters
    """
    
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = RoPEAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.ModuleDict(dict(
            c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=True),
            c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=True),
            dropout = nn.Dropout(config.resid_pdrop),
        ))
        m = self.mlp
        self.mlpf = lambda x: m.dropout(m.c_proj(F.gelu(m.c_fc(x))))  # MLP forward
    
    def forward(self, x, attention_mask=None):
        x = x + self.attn(self.ln_1(x), attention_mask=attention_mask)
        x = x + self.mlpf(self.ln_2(x))
        return x


class RoPEGPT(nn.Module):
    """
    GPT model with Rotary Position Embedding (RoPE).
    
    This is a complete GPT-style transformer model that uses RoPE for position
    encoding instead of learnable position embeddings. The model architecture
    is similar to GPT-2 but without position embeddings, as position information
    is encoded directly in the attention mechanism through RoPE.
    
    Key differences from traditional GPT:
    - No position embeddings (wpe) - position info encoded via RoPE
    - Uses RoPEAttention in transformer blocks
    - Better extrapolation to longer sequences
    
    Args:
        config: Configuration object containing all model parameters
    """
    
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.block_size = config.block_size
        self.config = config
        
        # Token and position embeddings
        vocab_size = config.vocab_size[0] if isinstance(config.vocab_size, list) else config.vocab_size
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(vocab_size, config.n_embd),
            # No position embedding needed for RoPE
            drop = nn.Dropout(config.embd_pdrop),
            h = nn.ModuleList([RoPETransformerBlock(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, vocab_size, bias=False)
        
        # Weight sharing
        self.transformer.wte.weight = self.lm_head.weight
        
        # Init weights
        self.apply(self._init_weights)
        
        # Report number of parameters
        n_params = sum(p.numel() for p in self.parameters())
        print(f"RoPE GPT model with {n_params/1e6:.2f}M parameters")
    
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
    
    def forward(self, idx, targets=None, attention_mask=None):
        """
        Forward pass through the RoPE transformer model.
        
        Args:
            idx (torch.Tensor): Input token indices of shape (batch_size, sequence_length)
            targets (torch.Tensor, optional): Target token indices for loss calculation
            attention_mask (torch.Tensor, optional): Attention mask for padded sequences
            
        Returns:
            torch.Tensor or tuple: If targets is None, returns logits of shape (batch_size, sequence_length, vocab_size).
                                  If targets is provided, returns (logits, loss).
        """
        device = idx.device
        b, t = idx.size()
        assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"
        
        # Token embeddings
        tok_emb = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)
        x = self.transformer.drop(tok_emb)
        
        # Forward through transformer blocks
        for block in self.transformer.h:
            x = block(x, attention_mask=attention_mask)
        
        x = self.transformer.ln_f(x)
        
        if targets is not None:
            # If we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # Inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :])  # note: using list [-1] to preserve the time dim
            loss = None
        
        return logits, loss
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, do_sample=False, top_k=None, end_token=None):
        """
        Generate text by auto-regressively sampling from the model.
        
        Args:
            idx (torch.Tensor): Conditioning sequence of token indices, shape (batch_size, sequence_length)
            max_new_tokens (int): Maximum number of new tokens to generate
            temperature (float): Sampling temperature (1.0 = no change, >1.0 = more random, <1.0 = more deterministic)
            do_sample (bool): If True, sample from probability distribution. If False, use greedy decoding
            top_k (int, optional): If specified, only consider top k tokens for sampling
            end_token (int, optional): Token ID to stop generation early (not implemented in current version)
            
        Returns:
            torch.Tensor: Generated sequence including the original conditioning sequence,
                         shape (batch_size, original_length + generated_length)
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # either sample from the distribution or take the most likely element
            if do_sample:
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                _, idx_next = torch.topk(probs, k=1, dim=-1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)
            # Stop if we hit the end token
            if end_token is not None and idx_next.item() == end_token:
                break
                
        return idx



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
                betas=train_config.betas
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
                betas=train_config.betas
            )
        
        return optimizer

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
            
            # Use our custom RoPE generate method
            generated_ids = self.model.generate(
                input_ids, 
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                top_k=top_k,
                end_token=end_token
            )
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