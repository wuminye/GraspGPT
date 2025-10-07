"""
Full definition of a GPT Language Model, all of it in this single file.

References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
from typing import Dict, List, Optional, Set

import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoConfig, Qwen2ForCausalLM, Qwen2Config
from tqdm.auto import tqdm

from .token_manager import get_token_manager
from .utils import CfgNode as CN


# -----------------------------------------------------------------------------



class GrammarHelper:
    """Token metadata helpers shared across grammar states."""

    def __init__(self, vocab_size: int):
        token_manager = get_token_manager()
        base_tokens = token_manager.all_tokens
        self.vocab_size = vocab_size
        self.coord_start = len(base_tokens)
        self.id_to_token: Dict[int, str] = {idx: tok for idx, tok in enumerate(base_tokens)}
        self.token_to_id: Dict[str, int] = {tok: idx for idx, tok in enumerate(base_tokens)}
        self.shape_tag_ids: Set[int] = {
            self.token_to_id[tag] for tag in token_manager.shape_tags if tag in self.token_to_id
        }
        self.object_tag_ids: Set[int] = {
            idx for tag, idx in self.token_to_id.items()
            if tag.startswith('object') and tag[6:].isdigit()
        }
        self.serial_ids: Set[int] = {
            self.token_to_id[token]
            for token in token_manager.serial_tokens
            if token in self.token_to_id
        }
        self.scene_id: Optional[int] = self.token_to_id.get('scene')
        self.amodal_id: Optional[int] = self.token_to_id.get('amodal')
        self.endamodal_id: Optional[int] = self.token_to_id.get('endamodal')
        self.segment_id: Optional[int] = self.token_to_id.get('segment')
        self.endunseg_id: Optional[int] = self.token_to_id.get('endunseg')
        self.detectgrasp_id: Optional[int] = self.token_to_id.get('detectgrasp')
        self.grasp_id: Optional[int] = self.token_to_id.get('grasp')
        self.end_id: Optional[int] = self.token_to_id.get('end')
        self.unlabel_id: Optional[int] = self.token_to_id.get('unlabel')

    def is_coord(self, token_id: int) -> bool:
        return self.coord_start <= token_id < self.vocab_size


class GrammarState:
    """Track allowable tokens for one sequence prefix.

    Feature toggles (serial tokens, UNSEG sections, CB monotonicity) are
    injected by `graspGPT.generate` so the same logic can be reused for
    prompts with different grammar requirements.
    """

    def __init__(
        self,
        helper: GrammarHelper,
        *,
        allow_serial: bool,
        allow_unseg: bool,
        enforce_monotonic_cb: bool,
        grasp_cb_count: Optional[int],
    ):
        self.h = helper
        self.allow_serial = allow_serial
        self.allow_unseg = allow_unseg
        self.enforce_monotonic_cb = enforce_monotonic_cb
        self.grasp_cb_target = grasp_cb_count
        self.current_section: str = 'pre_scene'
        self.scene_started = False
        self.scene_done = False
        self.amodal_available = True
        self.segment_available = allow_unseg
        self.detectgrasp_available = True
        self.amodal_open = False
        self.amodal_sb_done = False
        self.segment_open = False
        self.detectgrasp_open = False
        self.pending_grasp_tag = False
        self.inside_sb = False
        self.sb_context: Optional[str] = None
        self.sb_has_coord = False
        self.just_saw_coord = False
        self.last_coord_id: Optional[int] = None
        self.sb_coord_count = 0
        self.finished = False

    def consume(self, token_id: int) -> None:
        if self.finished:
            return

        if self.inside_sb and not (self.h.is_coord(token_id) or token_id in self.h.serial_ids):
            self._exit_sb()

        if self.h.is_coord(token_id):
            self._consume_coord(token_id)
            return

        if token_id in self.h.serial_ids:
            self._consume_serial()
            return

        token = self.h.id_to_token.get(token_id)

        if token_id in self.h.shape_tag_ids:
            self._consume_shape_tag(token_id, token)
        else:
            self._consume_command(token_id, token)

    def build_mask(self, device: torch.device) -> torch.Tensor:
        mask = torch.zeros(self.h.vocab_size, dtype=torch.bool, device=device)

        if self.finished:
            if self.h.end_id is not None:
                mask[self.h.end_id] = True
            return mask

        if self.pending_grasp_tag:
            for idx in self.h.shape_tag_ids:
                mask[idx] = True
            return mask

        if self.inside_sb:
            start_idx = self.h.coord_start
            if self.enforce_monotonic_cb and self.last_coord_id is not None:
                start_idx = max(start_idx, self.last_coord_id + 1)
            allow_more_coords = True
            if self.sb_context == 'grasp' and self.grasp_cb_target is not None:
                allow_more_coords = self.sb_coord_count < self.grasp_cb_target
            if allow_more_coords:
                mask[start_idx:self.h.vocab_size] = True
            if self.allow_serial and self.just_saw_coord:
                for idx in self.h.serial_ids:
                    mask[idx] = True
            if self.sb_has_coord:
                outer_tokens = self._outer_tokens_for_current_context()
                if self.sb_context == 'grasp' and self.grasp_cb_target is not None and self.sb_coord_count < self.grasp_cb_target:
                    outer_tokens = set()
                for idx in outer_tokens:
                    if idx is not None:
                        mask[idx] = True
            return mask

        if self.current_section == 'pre_scene':
            if self.h.scene_id is not None:
                mask[self.h.scene_id] = True
            return mask

        if self.current_section == 'scene':
            for idx in self.h.shape_tag_ids:
                mask[idx] = True
            for idx in self._allowed_sections_after_scene():
                mask[idx] = True
            return mask

        if self.current_section == 'between_sections':
            for idx in self._allowed_sections_after_scene():
                mask[idx] = True
            return mask

        if self.current_section == 'amodal':
            if not self.amodal_sb_done:
                if self.h.unlabel_id is not None:
                    mask[self.h.unlabel_id] = True
            else:
                if self.h.endamodal_id is not None:
                    mask[self.h.endamodal_id] = True
            return mask

        if self.current_section == 'segment':
            if self.allow_unseg:
                for idx in self.h.object_tag_ids:
                    mask[idx] = True
            if self.h.endunseg_id is not None:
                mask[self.h.endunseg_id] = True
            return mask

        if self.current_section == 'detectgrasp':
            if self.h.grasp_id is not None:
                mask[self.h.grasp_id] = True
            if self.h.end_id is not None:
                mask[self.h.end_id] = True
            return mask

        # Fallback, should not reach here during valid generation
        if self.h.end_id is not None:
            mask[self.h.end_id] = True
        return mask

    def _consume_coord(self, token_id: int) -> None:
        if not self.inside_sb:
            return
        if self.enforce_monotonic_cb:
            if self.last_coord_id is None:
                self.last_coord_id = token_id
            else:
                self.last_coord_id = max(self.last_coord_id, token_id)
        self.sb_has_coord = True
        self.just_saw_coord = True
        self.sb_coord_count += 1

    def _consume_serial(self) -> None:
        if not self.inside_sb:
            return
        self.just_saw_coord = False

    def _consume_shape_tag(self, token_id: int, token: Optional[str]) -> None:
        if self.pending_grasp_tag:
            self.pending_grasp_tag = False
            self._enter_sb('grasp')
            return

        if self.current_section == 'scene':
            self._enter_sb('scene')
        elif self.current_section == 'amodal':
            self._enter_sb('amodal')
        elif self.current_section == 'segment':
            self._enter_sb('segment')
        elif self.current_section == 'detectgrasp':
            self._enter_sb('grasp')

    def _consume_command(self, token_id: int, token: Optional[str]) -> None:
        if token_id == self.h.scene_id:
            self.current_section = 'scene'
            self.scene_started = True
            return

        if token_id == self.h.amodal_id:
            self.scene_done = True
            self.current_section = 'amodal'
            self.amodal_open = True
            self.amodal_sb_done = False
            self.amodal_available = False
            return

        if token_id == self.h.endamodal_id:
            self.amodal_open = False
            self.amodal_sb_done = False
            self.current_section = 'between_sections'
            return

        if token_id == self.h.segment_id:
            self.scene_done = True
            self.current_section = 'segment'
            self.segment_open = True
            self.segment_available = False
            self.amodal_available = False
            return

        if token_id == self.h.endunseg_id:
            self.segment_open = False
            self.current_section = 'between_sections'
            return

        if token_id == self.h.detectgrasp_id:
            self.scene_done = True
            self.current_section = 'detectgrasp'
            self.detectgrasp_open = True
            self.detectgrasp_available = False
            self.amodal_available = False
            self.segment_available = False
            return

        if token_id == self.h.grasp_id:
            self.pending_grasp_tag = True
            return

        if token_id == self.h.end_id:
            self.scene_done = True
            self.current_section = 'finished'
            self.finished = True
            self.detectgrasp_open = False
            self.amodal_available = False
            self.segment_available = False
            self.detectgrasp_available = False

    def _enter_sb(self, context: str) -> None:
        self.inside_sb = True
        self.sb_context = context
        self.sb_has_coord = False
        self.just_saw_coord = False
        self.last_coord_id = None
        self.sb_coord_count = 0
        if context == 'amodal':
            self.amodal_open = True
        elif context == 'segment':
            self.segment_open = True
        elif context == 'grasp':
            self.detectgrasp_open = True

    def _exit_sb(self) -> None:
        context = self.sb_context
        self.inside_sb = False
        self.sb_context = None
        self.sb_has_coord = False
        self.just_saw_coord = False
        self.last_coord_id = None
        self.sb_coord_count = 0
        if context == 'amodal':
            self.amodal_sb_done = True
        elif context == 'grasp':
            self.detectgrasp_open = True

    def _outer_tokens_for_current_context(self) -> Set[int]:
        if self.sb_context == 'scene':
            tokens = set(self.h.shape_tag_ids)
            tokens.update(self._allowed_sections_after_scene())
            return tokens
        if self.sb_context == 'amodal':
            return {self.h.endamodal_id} if self.h.endamodal_id is not None else set()
        if self.sb_context == 'segment':
            if self.allow_unseg:
                tokens = set(self.h.object_tag_ids)
                if self.h.endunseg_id is not None:
                    tokens.add(self.h.endunseg_id)
                return tokens
            return {self.h.endunseg_id} if self.h.endunseg_id is not None else set()
        if self.sb_context == 'grasp':
            tokens: Set[int] = set()
            if self.h.grasp_id is not None:
                tokens.add(self.h.grasp_id)
            if self.h.end_id is not None:
                tokens.add(self.h.end_id)
            return tokens
        return set()

    def _allowed_sections_after_scene(self) -> Set[int]:
        tokens: Set[int] = set()
        if self.scene_started:
            if self.amodal_available and self.h.amodal_id is not None:
                tokens.add(self.h.amodal_id)
            if self.segment_available and self.h.segment_id is not None:
                tokens.add(self.h.segment_id)
            if self.detectgrasp_available and self.h.detectgrasp_id is not None:
                tokens.add(self.h.detectgrasp_id)
            if self.h.end_id is not None:
                tokens.add(self.h.end_id)
        return tokens

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
                'gpt2-small':   dict(n_layer=16, n_head=16, n_embd=1024),  # 124M params
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
                num_key_value_heads=4,  # Set key-value heads equal to attention heads
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

    @staticmethod
    def _compute_loss(logits, targets, loss_mask=None):
        target_tokens = targets[..., 0].long().reshape(-1)
        logits_flat = logits.reshape(-1, logits.size(-1))

        if loss_mask is not None:
            mask = loss_mask.reshape(-1).to(logits.device)
            if mask.dtype != torch.bool:
                mask = mask > 0
            valid = mask & (target_tokens != -1)
            if valid.any():
                return F.cross_entropy(logits_flat[valid], target_tokens[valid])

        non_padding = target_tokens != -1
        if non_padding.any():
            return F.cross_entropy(logits_flat[non_padding], target_tokens[non_padding])

        return logits_flat.sum() * 0.0

    def forward(self, idx, targets=None, attention_mask=None, loss_mask=None):
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

        logits_heads = [logits]
        loss = None
        if targets is not None:
            loss = self._compute_loss(logits, targets, loss_mask=loss_mask)

        return logits_heads, loss

    @torch.no_grad()
    def generate(
        self,
        idx,
        max_new_tokens,
        temperature=1.0,
        do_sample=False,
        top_k=None,
        end_token=None,
        show_progress=True,
        allow_serial=False,
        allow_unseg=False,
        enforce_monotonic_cb=True,
        grasp_cb_count: Optional[int] = 3,
    ):
        """Sample tokens sequentially while enforcing grammar constraints.

        Args:
            idx: Conditioning tokens.
            max_new_tokens: Number of tokens to generate.
            temperature: Sampling temperature.
            do_sample: Whether to sample instead of greedy.
            top_k: Optional top-k filtering.
            end_token: Optional EOS token id.
            show_progress: Display generation progress bar when True.
            allow_serial: Enable SERIAL tokens after coordinates (disabled by default).
            allow_unseg: Enable UNSEG sections in generated output (disabled by default).
            enforce_monotonic_cb: Require CB coordinates to be strictly increasing.
            grasp_cb_count: Require each GB to emit exactly this many CB blocks (None disables).
        """

        if max_new_tokens <= 0:
            if idx.dim() > 2:
                return idx
            return idx.unsqueeze(-1)

        input_ids = idx[..., 0].long() if idx.dim() > 2 else idx.long()
        device = next(self.model.parameters()).device
        input_ids = input_ids.to(device)

        vocab_size = getattr(self.model.config, 'vocab_size', None)
        if vocab_size is None:
            cfg_vocab = self.config.vocab_size
            vocab_size = cfg_vocab[0] if isinstance(cfg_vocab, (list, tuple)) else cfg_vocab
        helper = GrammarHelper(vocab_size)

        batch_size = input_ids.size(0)
        # Propagate feature toggles to grammar tracking state for each sample.
        grammar_states = [
            GrammarState(
                helper,
                allow_serial=allow_serial,
                allow_unseg=allow_unseg,
                enforce_monotonic_cb=enforce_monotonic_cb,
                grasp_cb_count=grasp_cb_count,
            )
            for _ in range(batch_size)
        ]
        for b in range(batch_size):
            for token in input_ids[b].tolist():
                grammar_states[b].consume(int(token))

        end_token_id = end_token if end_token is not None else helper.end_id
        generated = input_ids
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        if end_token_id is not None and generated.size(1) > 0:
            finished |= generated[:, -1] == end_token_id

        past_key_values = None
        was_training = self.model.training
        self.model.eval()

        temp = float(max(temperature, 1e-5))

        progress_bar = tqdm(total=max_new_tokens, desc="Generating", disable=not show_progress, leave=False)
        try:
            for _ in range(max_new_tokens):
                if past_key_values is None:
                    model_inputs = generated
                else:
                    model_inputs = generated[:, -1:].contiguous()

                outputs = self.model(
                    input_ids=model_inputs,
                    use_cache=True,
                    past_key_values=past_key_values,
                    return_dict=True
                )

                logits = outputs.logits[:, -1, :]
                past_key_values = outputs.past_key_values
                if temp != 1.0:
                    logits = logits / temp

                next_tokens: List[int] = []

                for b in range(batch_size):
                    state = grammar_states[b]
                    if finished[b] and end_token_id is not None:
                        forced = int(end_token_id)
                        next_tokens.append(forced)
                        state.consume(forced)
                        continue

                    mask = state.build_mask(logits.device)
                    if mask.sum().item() == 0:
                        if end_token_id is not None and 0 <= end_token_id < mask.size(0):
                            mask[end_token_id] = True
                        else:
                            mask[:] = True

                    logits_b = logits[b].clone()
                    logits_b[~mask] = float('-inf')

                    if top_k is not None and top_k > 0:
                        valid_count = int(mask.sum().item())
                        k = min(top_k, valid_count)
                        if k > 0 and k < logits_b.size(-1):
                            _, topk_idx = torch.topk(logits_b, k)
                            new_mask = torch.zeros_like(mask)
                            new_mask[topk_idx] = True
                            logits_b[~new_mask] = float('-inf')
                            mask = new_mask

                    if not torch.isfinite(logits_b).any():
                        if end_token_id is not None:
                            logits_b[end_token_id] = 0.0
                        else:
                            fallback_idx = mask.nonzero(as_tuple=False)
                            idx_choice = int(fallback_idx[0].item()) if fallback_idx.numel() > 0 else 0
                            logits_b[idx_choice] = 0.0

                    if do_sample:
                        probs = torch.softmax(logits_b, dim=-1)
                        if torch.isnan(probs).any() or probs.sum().item() <= 0:
                            selected = int(torch.argmax(logits_b).item())
                        else:
                            selected = int(torch.multinomial(probs, num_samples=1).item())
                    else:
                        selected = int(torch.argmax(logits_b).item())

                    next_tokens.append(selected)
                    state.consume(selected)
                    if end_token_id is not None and selected == end_token_id:
                        finished[b] = True

                progress_bar.update(1)

                new_tokens = torch.tensor(next_tokens, dtype=torch.long, device=device).unsqueeze(1)
                generated = torch.cat([generated, new_tokens], dim=1)

                if finished.all():
                    break
        finally:
            progress_bar.close()

        if was_training:
            self.model.train()

        if idx.dim() > 2:
            expanded = generated.unsqueeze(-1).expand(-1, -1, idx.size(-1))
            return expanded
        return generated.unsqueeze(-1)

    @torch.no_grad()
    def generate_ori(self, idx, max_new_tokens, temperature=1.0, do_sample=False, top_k=None, end_token=None, **kwargs):
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
