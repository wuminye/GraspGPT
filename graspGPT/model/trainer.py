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





def pad_collate(batch):
    """
    Pads the batch of (x, y, attention_mask) tuples to the same temporal length.
    Handles variable sequence lengths by padding to the maximum length in the batch.
    
    Args:
        batch: List of tuples (input_tokens, target_tokens, attention_mask)
            - input_tokens: tensor of shape [seq_len, num_features] 
            - target_tokens: tensor of shape [seq_len, num_features]
            - attention_mask: tensor of shape [seq_len]
    
    Returns:
        x_batch: (batch_size, max_seq_len, num_features) - padded input tokens
        y_batch: (batch_size, max_seq_len, num_features) - padded target tokens  
        att_batch: (batch_size, max_seq_len) - padded attention masks
    """
    xs, ys, atts = zip(*batch)
    
    # Find maximum sequence length in the batch
    max_length = max(x.shape[0] for x in xs)
    batch_size = len(xs)
    
    # Get number of features from first sample
    num_features = xs[0].shape[1] if xs[0].dim() > 1 else 1
    
    # Initialize padded tensors
    x_batch = torch.zeros(batch_size, max_length, num_features, dtype=xs[0].dtype)
    y_batch = torch.zeros(batch_size, max_length, num_features, dtype=ys[0].dtype)
    att_batch = torch.zeros(batch_size, max_length, dtype=atts[0].dtype)
    
    # Fill in the actual data
    for i, (x, y, att) in enumerate(zip(xs, ys, atts)):
        seq_len = x.shape[0]
        if x.dim() == 1:
            # Handle 1D case by adding feature dimension
            x = x.unsqueeze(-1)
            y = y.unsqueeze(-1)
        
        x_batch[i, :seq_len] = x
        y_batch[i, :seq_len] = y  
        y_batch[i, seq_len:] = -1
        att_batch[i, :seq_len] = att

    return x_batch, y_batch, att_batch

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
            x, y, att = batch  
            if self.use_amp:
                with autocast('cuda'):
                    logits, self.loss = model(x, targets=y, attention_mask=att)
                model.zero_grad(set_to_none=True)
                self.scaler.scale(self.loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits, self.loss = model(x, targets=y, attention_mask=att)
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

