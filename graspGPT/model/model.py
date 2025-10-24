"""
Full definition of a GPT Language Model, all of it in this single file.

References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple, Union
import bisect

import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoConfig, Qwen2ForCausalLM, Qwen2Config
from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList
from tqdm.auto import tqdm

from .token_manager import get_token_manager


Coord = Tuple[int, int, int]
Token = Union[str, Coord]


class GrammarConstraintViolation(Exception):
    """Raised when a token sequence violates the scene grammar."""


@dataclass
class BlockState:
    kind: str
    expecting: str
    last_coord: Optional[Coord] = None


class GrammarState:
    """Tracks grammar progress and computes valid next token ids."""

    def __init__(
        self,
        token_mapping: Dict[Token, int],
        token_manager=None,
        ignore_amodal: bool = True,
        end_token_id: Optional[int] = None,
    ) -> None:
        self.token_manager = token_manager or get_token_manager()
        self.token_mapping = token_mapping
        self.id_to_token: Dict[int, Token] = {v: k for k, v in token_mapping.items()}
        self.ignore_amodal = ignore_amodal

        self.scene_token_id = token_mapping.get('scene')
        self.segment_token_id = token_mapping.get('segment')
        self.endunseg_token_id = token_mapping.get('endunseg')
        self.detectgrasp_token_id = token_mapping.get('detectgrasp')
        self.grasp_token_id = token_mapping.get('grasp')
        self.end_token_id = end_token_id if end_token_id is not None else token_mapping.get('end')
        if self.scene_token_id is None or self.end_token_id is None:
            raise ValueError("token_mapping must contain 'scene' and 'end' tokens.")

        self._shape_tag_ids: Set[int] = self._collect_ids(self.token_manager.shape_tags)
        self._object_tag_ids: Set[int] = {
            token_mapping[tag]
            for tag in self.token_manager.shape_tags
            if tag.startswith('object') and tag in token_mapping
        }
        coord_items = [
            (token, idx)
            for token, idx in token_mapping.items()
            if isinstance(token, tuple)
            and len(token) == 3
            and all(isinstance(n, int) for n in token)
        ]
        coord_items.sort(key=lambda pair: pair[0])
        self._coord_coords_sorted: List[Coord] = [coord for coord, _ in coord_items]
        self._coord_ids_sorted: List[int] = [idx for _, idx in coord_items]

        self.phase = 'pre_scene'
        self.current_block: Optional[BlockState] = None
        self.unseg_started = False
        self.grasp_started = False

    def _collect_ids(self, tokens: List[str]) -> Set[int]:
        return {self.token_mapping[token] for token in tokens if token in self.token_mapping}

    @staticmethod
    def _is_coord(token: Token) -> bool:
        return isinstance(token, tuple) and len(token) == 3 and all(isinstance(n, int) for n in token)

    @staticmethod
    def _is_object_tag(token: Token) -> bool:
        return isinstance(token, str) and token.startswith('object') and token[6:].isdigit()

    @staticmethod
    def _coord_greater(a: Coord, b: Coord) -> bool:
        return a > b

    def _iter_coordinate_ids(self, last_coord: Optional[Coord], enforce_order: bool) -> List[int]:
        if not self._coord_ids_sorted:
            return []
        if not enforce_order or last_coord is None:
            return self._coord_ids_sorted
        idx = bisect.bisect(self._coord_coords_sorted, last_coord)
        if idx >= len(self._coord_ids_sorted):
            return []
        return self._coord_ids_sorted[idx:]

    def process_token(self, token: Token) -> None:
        if self.ignore_amodal and isinstance(token, str) and token in {'amodal', 'endamodal'}:
            raise GrammarConstraintViolation("Amodal tokens are not supported in constrained generation.")

        if self.phase == 'pre_scene':
            if token != 'scene':
                raise GrammarConstraintViolation("Sequence must start with 'scene'.")
            self.phase = 'scene'
            return

        if self.current_block is not None:
            consumed = self._process_block_token(token)
            if consumed:
                return

        self._process_outer_token(token)

    def _process_block_token(self, token: Token) -> bool:
        block = self.current_block
        if block is None:
            return False

        if block.kind == 'grasp_gb' and block.expecting == 'expect_tag':
            if not (isinstance(token, str) and self.token_manager.is_shape_tag(token)):
                raise GrammarConstraintViolation("GRASP block requires a valid shape tag after 'grasp'.")
            block.expecting = 'expect_coord'
            block.last_coord = None
            return True

        if block.expecting == 'expect_coord':
            if not self._is_coord(token):
                raise GrammarConstraintViolation("At least one coordinate is required after a tag.")
            #self._validate_coord(block, token)
            block.last_coord = token
            block.expecting = 'after_coord'
            return True

        if block.expecting == 'after_coord':
            if isinstance(token, str) and self.token_manager.is_serial_token(token):
                raise GrammarConstraintViolation("Serial tokens are not supported.")
            if self._is_coord(token):
                #self._validate_coord(block, token)
                block.last_coord = token
                block.expecting = 'after_coord'
                return True
            self.current_block = None
            return False

        raise GrammarConstraintViolation(f"Unknown block state: {block.expecting}")

    def _validate_coord(self, block: BlockState, coord: Coord) -> None:
        if block.kind in {'scene_sb', 'unseg_sb'} and block.last_coord is not None:
            if not self._coord_greater(coord, block.last_coord):
                raise GrammarConstraintViolation("Coordinates inside SB must be strictly increasing.")

    def _process_outer_token(self, token: Token) -> None:
        if isinstance(token, tuple):
            raise GrammarConstraintViolation("Coordinate token encountered outside of a block.")

        if isinstance(token, str) and self.token_manager.is_serial_token(token):
            raise GrammarConstraintViolation("Serial tokens are not supported.")

        if self.phase == 'scene':
            if self.token_manager.is_shape_tag(token):
                self.current_block = BlockState(kind='scene_sb', expecting='expect_coord')
                return
            if token == 'segment':
                if self.unseg_started:
                    raise GrammarConstraintViolation("UNSEG section cannot appear twice.")
                self.unseg_started = True
                self.phase = 'unseg'
                return
            if token == 'detectgrasp':
                if self.grasp_started:
                    raise GrammarConstraintViolation("GRASP section cannot appear twice.")
                self.grasp_started = True
                self.phase = 'grasp'
                return
            if token == 'end':
                self.phase = 'done'
                return
            raise GrammarConstraintViolation(f"Unexpected token in scene: {token!r}")

        if self.phase == 'unseg':
            if self._is_object_tag(token) and self.token_manager.is_shape_tag(token):
                self.current_block = BlockState(kind='unseg_sb', expecting='expect_coord')
                return
            if token == 'endunseg':
                self.phase = 'post_unseg'
                return
            raise GrammarConstraintViolation(f"Unexpected token in UNSEG: {token!r}")

        if self.phase == 'post_unseg':
            if token == 'detectgrasp':
                if self.grasp_started:
                    raise GrammarConstraintViolation("GRASP section cannot appear twice.")
                self.grasp_started = True
                self.phase = 'grasp'
                return
            if token == 'end':
                self.phase = 'done'
                return
            raise GrammarConstraintViolation(f"Unexpected token after UNSEG: {token!r}")

        if self.phase == 'grasp':
            if token == 'grasp':
                self.current_block = BlockState(kind='grasp_gb', expecting='expect_tag')
                return
            if token == 'end':
                self.phase = 'done'
                return
            raise GrammarConstraintViolation(f"Unexpected token in GRASP: {token!r}")

        if self.phase == 'done':
            if token == 'end':
                return
            raise GrammarConstraintViolation("No tokens allowed after 'end'.")

        raise GrammarConstraintViolation(f"Unhandled phase: {self.phase}")

    def allowed_token_ids(self) -> List[int]:
        if self.phase == 'pre_scene':
            return [self.scene_token_id]  # type: ignore[list-item]

        if self.current_block is not None:
            block = self.current_block
            allowed: Set[int] = set()
            if block.kind == 'grasp_gb' and block.expecting == 'expect_tag':
                allowed.update(self._shape_tag_ids)
                return sorted(allowed)

            enforce_order = block.kind in {'scene_sb', 'unseg_sb'}
            enforce_order = False
            if block.expecting == 'expect_coord':
                allowed.update(self._iter_coordinate_ids(block.last_coord, enforce_order))
                return sorted(allowed)

            if block.expecting == 'after_coord':
                allowed.update(self._iter_coordinate_ids(block.last_coord, enforce_order))
                allowed.update(self._allowed_tokens_no_block())
                return sorted(allowed)

            raise GrammarConstraintViolation(f"Unknown block state: {block.expecting}")

        return sorted(self._allowed_tokens_no_block())

    def _allowed_tokens_no_block(self) -> Set[int]:
        allowed: Set[int] = set()
        if self.phase == 'scene':
            allowed.update(self._shape_tag_ids)
            if not self.unseg_started and self.segment_token_id is not None:
                allowed.add(self.segment_token_id)
            if not self.grasp_started and self.detectgrasp_token_id is not None:
                allowed.add(self.detectgrasp_token_id)
            if self.end_token_id is not None:
                allowed.add(self.end_token_id)
            return allowed

        if self.phase == 'unseg':
            allowed.update(self._object_tag_ids)
            if self.endunseg_token_id is not None:
                allowed.add(self.endunseg_token_id)
            return allowed

        if self.phase == 'post_unseg':
            if not self.grasp_started and self.detectgrasp_token_id is not None:
                allowed.add(self.detectgrasp_token_id)
            if self.end_token_id is not None:
                allowed.add(self.end_token_id)
            return allowed

        if self.phase == 'grasp':
            if self.grasp_token_id is not None:
                allowed.add(self.grasp_token_id)
            if self.end_token_id is not None:
                allowed.add(self.end_token_id)
            return allowed

        if self.phase == 'done' and self.end_token_id is not None:
            allowed.add(self.end_token_id)
            return allowed

        return allowed


class GrammarConstraintLogitsProcessor(LogitsProcessor):
    """Logits processor that enforces GrammarState constraints step-by-step."""

    def __init__(
        self,
        token_mapping: Dict[Token, int],
        ignore_amodal: bool = True,
        token_manager=None,
        end_token_id: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.token_mapping = token_mapping
        self.token_manager = token_manager or get_token_manager()
        self.ignore_amodal = ignore_amodal
        self.id_to_token: Dict[int, Token] = {v: k for k, v in token_mapping.items()}
        self.end_token_id = end_token_id if end_token_id is not None else token_mapping.get('end')
        self._states: Dict[int, GrammarState] = {}
        self._processed_lengths: Dict[int, int] = {}

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        batch_size = input_ids.size(0)
        for batch_idx in range(batch_size):
            '''
            state = self._states.get(batch_idx)
            if state is None:
                state = GrammarState(
                    token_mapping=self.token_mapping,
                    token_manager=self.token_manager,
                    ignore_amodal=self.ignore_amodal,
                    end_token_id=self.end_token_id,
                )
                self._states[batch_idx] = state
                self._processed_lengths[batch_idx] = 0

            processed = self._processed_lengths[batch_idx]
            sequence = input_ids[batch_idx].tolist()
            if len(sequence) < processed:
                state = GrammarState(
                    token_mapping=self.token_mapping,
                    token_manager=self.token_manager,
                    ignore_amodal=self.ignore_amodal,
                    end_token_id=self.end_token_id,
                )
                self._states[batch_idx] = state
                processed = 0

            for token_id in sequence[processed:]:
                if token_id < 0:
                    continue
                token = self.id_to_token.get(int(token_id))
                if token is None:
                    raise GrammarConstraintViolation(f"Token id {token_id} not found in mapping.")
                state.process_token(token)

            self._processed_lengths[batch_idx] = len(sequence)

            allowed = state.allowed_token_ids()
            if not allowed:
                raise GrammarConstraintViolation("No valid next tokens available under grammar constraints.")

            mask = torch.ones_like(scores[batch_idx], dtype=torch.bool)
            mask[allowed] = False
            scores[batch_idx][mask] = -float('inf')
            '''
            scores[batch_idx][self.token_mapping['scene']] = -float('inf')

        return scores
from .utils import CfgNode as CN


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
                #MY models
                'gpt2-shallow-wide':    dict(n_layer=6, n_head=20, n_embd=1280),  
                'gpt2-shallow-wide-1600-25':    dict(n_layer=7, n_head=24, n_embd=1536),
                'gpt2-shallow-wide-2048':    dict(n_layer=6, n_head=16, n_embd=2048), 
                'gpt2-shallow-wide-2048-32':    dict(n_layer=6, n_head=32, n_embd=2048),
                'gpt2-shallow-wide-4096':    dict(n_layer=6, n_head=16, n_embd=4096), 
                'gpt2-deep-narrow':    dict(n_layer=24, n_head=8, n_embd=256), 
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
            assert mask.dtype == torch.int, f"loss_mask should be int tensor, {mask.dtype}"

            content_mask = (mask == 1) & (target_tokens != -1)
            structure_mask = (mask == 2) & (target_tokens != -1)
            #valid = mask & (target_tokens != -1)
            content_loss = 0.0
            structure_loss = 0.0
            if content_mask.any():
                content_loss = F.cross_entropy(logits_flat[content_mask], target_tokens[content_mask])
            if structure_mask.any():
                structure_loss = F.cross_entropy(logits_flat[structure_mask], target_tokens[structure_mask])
            return content_loss, structure_loss

        non_padding = target_tokens != -1
        if non_padding.any():
            return F.cross_entropy(logits_flat[non_padding], target_tokens[non_padding]), logits_flat.sum() * 0.0

        return logits_flat.sum() * 0.0, logits_flat.sum() * 0.0

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
            loss, structure_loss = self._compute_loss(logits, targets, loss_mask=loss_mask)

        return logits_heads, loss, structure_loss

   

    @torch.no_grad()
    def generate(
        self,
        idx,
        max_new_tokens,
        token_mapping: Optional[Dict[Token, int]] = None,
        temperature: float = 1.0,
        do_sample: bool = False,
        top_k: Optional[int] = None,
        end_token: Optional[int] = None,
        ignore_amodal: bool = True,
        **kwargs,
    ):
        """Grammar-constrained generation with optional fallback to unconstrained mode."""

        if token_mapping is None:
            return self.generate_ori(
                idx=idx,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                top_k=top_k,
                end_token=end_token,
                **kwargs,
            )

        device = next(self.parameters()).device
        input_ids = idx[..., 0].long() if idx.dim() > 2 else idx.long()
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        input_ids = input_ids.to(device)

        eos_token_id = end_token if end_token is not None else token_mapping.get('end')
        if eos_token_id is None:
            raise ValueError("end_token must be provided or present in token_mapping.")

        self.model.eval()
        logits_processor = LogitsProcessorList(
            [
                GrammarConstraintLogitsProcessor(
                    token_mapping=token_mapping,
                    ignore_amodal=ignore_amodal,
                    token_manager=get_token_manager(),
                    end_token_id=eos_token_id,
                )
            ]
        )

        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "do_sample": do_sample,
            "return_dict_in_generate": True,
            "output_scores": True,
            "use_cache": True,
            "logits_processor": logits_processor,
            "eos_token_id": eos_token_id,
            "pad_token_id": -1,
        }
        if top_k is not None:
            generation_kwargs["top_k"] = top_k
        generation_kwargs.update(kwargs)

        generated = self.model.generate(input_ids=input_ids, **generation_kwargs)
        sequences = generated.sequences

        if idx.dim() > 2:
            expanded = sequences.unsqueeze(-1).expand(-1, -1, idx.size(-1))
            return expanded
        return sequences.unsqueeze(-1)

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
