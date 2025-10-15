"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.
"""

import time
from collections import defaultdict

import torch
from torch.utils.data.dataloader import DataLoader
from .utils import CfgNode as CN
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ExponentialLR
from .token_manager import get_token_manager


_TOKEN_MANAGER = get_token_manager()
_TOKEN_TO_ID = {token: idx for idx, token in enumerate(_TOKEN_MANAGER.all_tokens)}
_COORDINATE_START_ID = len(_TOKEN_MANAGER.all_tokens)
_SCENE_TOKEN_ID = _TOKEN_TO_ID.get('scene')
_SCENE_TERMINATOR_IDS = {
    _TOKEN_TO_ID.get(token)
    for token in ('segment', 'amodal', 'detectgrasp', 'end')
    if token in _TOKEN_TO_ID
}
_SHAPE_TAG_ID_TO_NAME = {
    _TOKEN_TO_ID[tag]: tag
    for tag in _TOKEN_MANAGER.shape_tags
    if tag in _TOKEN_TO_ID
}
_SERIAL_TOKEN_IDS = {
    _TOKEN_TO_ID[token]
    for token in _TOKEN_MANAGER.serial_tokens
    if token in _TOKEN_TO_ID
}
_INCOMPLETE_TOKEN_ID = _TOKEN_TO_ID.get('incomplete')







def pad_collate(batch):
    """
    Pads the batch of (x, y) pairs to the same temporal length.
    Assumes x and y are (t x g) tensors.
    Returns:
        x: (b x t x g)
        y: (b x t x g)
    """
    tokens, sequence_length, masks = zip(*batch)
    max_length = max(sequence_length)

    chunks = []
    loss_masks = []
    for i in range(len(tokens)):
        chunk = tokens[i]
        mask = masks[i]
        
        # If this is the last chunk and it's shorter than max_sequence_length, pad it
        if chunk.shape[0] < max_length:
            pad_size = max_length - chunk.shape[0]
            # Create padding tensor with -1 values, matching the shape of chunk except first dimension
            pad_shape = [pad_size] + list(chunk.shape[1:])
            padding = torch.full(pad_shape, -1, dtype=chunk.dtype, device=chunk.device)
            chunk = torch.cat([chunk, padding], dim=0)

            mask_padding = torch.zeros(pad_size, dtype=mask.dtype, device=mask.device)
            mask = torch.cat([mask, mask_padding], dim=0)
        

        chunks.append(chunk)
        loss_masks.append(mask)

    chunks = torch.stack(chunks, dim=0)  # Shape: [num_chunks, max_sequence_length, ...]
    loss_masks = torch.stack(loss_masks, dim=0)

    x_batch = chunks[:, :-1, ...]  # Input tokens
    y_batch = chunks[:, 1:, ...]   # Target tokens (next token prediction)
    loss_mask_batch = loss_masks[:, 1:]
    non_padding = (y_batch[..., 0] != -1).float()
    loss_mask_batch = loss_mask_batch * non_padding

    x_batch[x_batch<0]=0


    return x_batch, y_batch, loss_mask_batch




def pad_collate_packed(batch):
    """
    Splits tokens by max_sequence_length into chunks and pads the last chunk if needed.
    
    Args:
        batch: List of tuples (tokens, max_sequence_length)
            - tokens: tensor of shape [seq_len, ...]
            - max_sequence_length: int, maximum sequence length for chunking
    
    Returns:
        List of tensors, where each tensor has shape [max_sequence_length, ...] except
        possibly the last one which is padded with -1 if it's shorter than max_sequence_length
    """
    tokens, max_sequence_length = zip(*batch)
    max_sequence_length = max_sequence_length[0]
    
    # Concatenate all tokens along the first dimension
    tokens = torch.cat(tokens, dim=0)
    
    # Split tokens into chunks of max_sequence_length
    seq_len = tokens.shape[0]
    chunks = []
    loss_masks = []

    for i in range(0, seq_len, max_sequence_length):
        chunk = tokens[i:i + max_sequence_length]
        mask = _build_incomplete_loss_mask(chunk)

        # If this is the last chunk and it's shorter than max_sequence_length, pad it
        if chunk.shape[0] < max_sequence_length:
            pad_size = max_sequence_length - chunk.shape[0]
            # Create padding tensor with -1 values, matching the shape of chunk except first dimension
            pad_shape = [pad_size] + list(chunk.shape[1:])
            padding = torch.full(pad_shape, -1, dtype=chunk.dtype, device=chunk.device)
            chunk = torch.cat([chunk, padding], dim=0)

            mask_padding = torch.zeros(pad_size, dtype=mask.dtype, device=mask.device)
            mask = torch.cat([mask, mask_padding], dim=0)

        chunks.append(chunk)
        loss_masks.append(mask)

    chunks = torch.stack(chunks, dim=0)  # Shape: [num_chunks, max_sequence_length, ...]
    loss_masks = torch.stack(loss_masks, dim=0)

    x_batch = chunks[:, :-1, ...]  # Input tokens
    y_batch = chunks[:, 1:, ...]   # Target tokens (next token prediction)
    loss_mask_batch = loss_masks[:, 1:]
    non_padding = (y_batch[..., 0] != -1).float()
    loss_mask_batch = loss_mask_batch * non_padding

    x_batch[x_batch<0]=0


    return x_batch, y_batch, loss_mask_batch

class Trainer:

    @staticmethod
    def get_default_config():
        C = CN()
        # device to train on
        C.device = 'auto'
        # dataloder parameters
        C.num_workers = 4
        # optimizer parameters
        C.max_iters = None
        C.batch_size = 64
        C.learning_rate = 3e-4
        C.betas = (0.9, 0.95)
        C.weight_decay = 0.1 # only applied on matmul weights
        C.grad_norm_clip = 1.0
        C.soft_loss = False
        C.sigma = 0.1
        # mixed precision training
        C.use_amp = False
        # learning rate scheduler
        C.scheduler_type = 'cosine'  # 'cosine', 'step', 'exponential', 'none'
        C.warmup_iters = 0  # number of warmup iterations
        C.min_lr = 2e-6  # minimum learning rate
        C.step_size = 1000  # for step scheduler
        C.gamma = 0.9  # for step and exponential scheduler
        return C

    def __init__(self, config, model, train_loader):
        self.config = config
        self.model = model
        self.optimizer = None
        self.scheduler = None
        self.train_loader = train_loader
        self.callbacks = defaultdict(list)
        # 不再在此处分配device，模型和dataloader已由外部处理
        self.use_amp = config.use_amp and torch.cuda.is_available()
        self.scaler = GradScaler('cuda') if self.use_amp else None
        self.iter_num = 0
        self.iter_time = 0.0
        self.iter_dt = 0.0

    def add_callback(self, onevent: str, callback):
        self.callbacks[onevent].append(callback)

    def set_callback(self, onevent: str, callback):
        self.callbacks[onevent] = [callback]

    def trigger_callbacks(self, onevent: str):
        for callback in self.callbacks.get(onevent, []):
            callback(self)

    def _setup_scheduler(self):
        """Setup learning rate scheduler based on config"""
        if self.config.scheduler_type == 'none' or self.config.max_iters is None:
            self.scheduler = None
            return
            
        if self.config.scheduler_type == 'cosine':
            self.scheduler = CosineAnnealingLR(
                self.optimizer, 
                T_max=self.config.max_iters,
                eta_min=self.config.min_lr
            )
        elif self.config.scheduler_type == 'step':
            self.scheduler = StepLR(
                self.optimizer,
                step_size=self.config.step_size,
                gamma=self.config.gamma
            )
        elif self.config.scheduler_type == 'exponential':
            self.scheduler = ExponentialLR(
                self.optimizer,
                gamma=self.config.gamma
            )
        else:
            raise ValueError(f"Unknown scheduler type: {self.config.scheduler_type}")

    def _get_lr(self):
        """Get current learning rate"""
        if self.optimizer is None:
            return 0.0
        return self.optimizer.param_groups[0]['lr']

    def _update_lr_with_warmup(self):
        """Update learning rate with warmup"""
        if self.config.warmup_iters > 0 and self.iter_num < self.config.warmup_iters:
            # Linear warmup
            lr_scale = min(1.0, self.iter_num / self.config.warmup_iters)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.config.learning_rate * lr_scale
        elif self.scheduler is not None:
            # Use scheduler after warmup
            self.scheduler.step()

    def run(self):
        model, config = self.model, self.config
        self.optimizer = model.module.configure_optimizers(config) if hasattr(model, 'module') else model.configure_optimizers(config)
        self._setup_scheduler()
        train_loader = self.train_loader
        model.train()
        self.iter_num = 0
        self.iter_time = time.time()
        data_iter = iter(train_loader)
        while True:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)
            batch = [t.cuda(non_blocking=True) if t is not None and torch.is_tensor(t) else t for t in batch]
            x, y, loss_mask = batch  

            logits, self.loss = model(x, targets=y, attention_mask=None, loss_mask=loss_mask)
            model.zero_grad(set_to_none=True)
            self.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
            self.optimizer.step()
            
            # Update learning rate with warmup and scheduler
            self._update_lr_with_warmup()
            
            self.trigger_callbacks('on_batch_end')
            self.iter_num += 1
            tnow = time.time()
            self.iter_dt = tnow - self.iter_time
            self.iter_time = tnow

