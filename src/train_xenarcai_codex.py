"""
training script for codex datasets

to run on a single gpu:
$ python train.py --batch_size=32 --compile=False

to run with ddp on 4 gpus on 1 node:
$ torchrun --standalone --nproc_per_node=4 train.py
"""

import os
import time
import math
import pickle
from contextlib import nullcontext
import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel
from torch.distributed import init_process_group, destroy_process_group
from trainer_model import Config, Module
from loader_xenarcai_codex import create_loader

# * configuration
out_dir = '.local/output'
eval_interval = 1024
log_interval = 1024
eval_only = False
always_save_checkpoint = True
init_from = 'scratch'
debug = False

# * wandb logging
wandb_log = False
wandb_project = 'nano'
wandb_run_name = 'nano-run'

# * data
gradient_accumulation_steps = 8
batch_size = 6
block_size = 6144

# * baby gpt model
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2
bias = False

# * adamw optimizer
learning_rate = 1e-3
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.99
grad_clip = 1.0

# * learning rate decay settings
decay_lr = True
warmup_iters = 100
min_lr = 1e-4

# * ddp settings
backend = 'nccl'

# * system
device = 'cuda'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compilation = True

# * various inits, derived attributes, i/o setup
ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
    seed_offset = ddp_rank
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    master_process = True
    seed_offset = 0
    ddp_world_size = 1

if master_process:
    os.makedirs(out_dir, exist_ok=True)

torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.fp32_precision = 'tf32'
torch.backends.cudnn.conv.fp32_precision = 'tf32'
device_type = 'cuda' if 'cuda' in device else 'cpu'
torch_dtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=torch_dtype)


# * data loader
def get_batch(loader):
    """get a batch from loader - one get() call per batch item"""
    x_list = []
    y_list = []

    for _ in range(batch_size):
        tokens = loader.get()

        # * reset loader if exhausted
        if tokens is None:
            loader.reset()
            tokens = loader.get()
            if tokens is None:
                raise ValueError("loader is empty")

        # * pad or truncate to block_size + 1
        if len(tokens) < block_size + 1:
            # * pad with zeros
            padded = np.zeros(block_size + 1, dtype=np.uint16)
            padded[:len(tokens)] = tokens
            tokens = padded
        elif len(tokens) > block_size + 1:
            # * truncate
            tokens = tokens[:block_size + 1]

        # * split into x and y
        x = torch.from_numpy(tokens[:block_size].astype(np.int64))
        y = torch.from_numpy(tokens[1:block_size + 1].astype(np.int64))

        x_list.append(x)
        y_list.append(y)

    # * stack into batches
    x = torch.stack(x_list)
    y = torch.stack(y_list)

    if device_type == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)

    return x, y


# * initialize loaders
if master_process:
    print("initializing loaders...")
loader = create_loader()

# * init these up here
iter_num = 0
best_val_loss = 1e9

# * get vocab size
meta_vocab_size = loader.nano.get_num()

if master_process:
    print(f"using vocab_size {meta_vocab_size}")

# * calculate max_iters based on dataset size
total_train_sequences = loader.total_items
sequences_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size
max_iters = total_train_sequences // sequences_per_iter
lr_decay_iters = max_iters

if master_process:
    print(f"total train sequences: {total_train_sequences:,}")
    print(f"sequences per iteration: {sequences_per_iter:,}")
    print(f"calculated max_iters: {max_iters:,}")

# * model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout)

if master_process:
    print("initializing a new model from scratch")
model_args['vocab_size'] = meta_vocab_size
config = Config(**model_args)
model = Module(config)

model.to(device)

# * initialize a gradscaler
scaler = torch.amp.GradScaler('cuda', enabled=(dtype == 'bfloat16'))

# * optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)

# * compile the model
if compilation:
    if master_process:
        print("compiling the model...")
    unoptimized_model = model
    model = torch.compile(model)

# * wrap model into ddp container
if ddp:
    model = DistributedDataParallel(model, device_ids=[ddp_local_rank])


# * estimate loss
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_interval)
        for k in range(eval_interval):
            X, Y = get_batch(split, loader)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# * learning rate decay scheduler
def get_lr(it):
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


# * logging
if wandb_log and master_process:
    import wandb

    wandb.init(project=wandb_project, name=wandb_run_name, config=None)

# * training loop
t0 = time.time()
local_iter_num = 0
raw_model = model.module if ddp else model
running_mfu = -1.0

if master_process:
    print("starting training loop...")

while True:
    # * set learning rate
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # * evaluate and save checkpoints
    if iter_num % eval_interval == 0 and master_process:
        # losses = estimate_loss()
        losses = {'train': 0.0, 'val': 0.0}
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "mfu": running_mfu * 100,
            })
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))

    if iter_num == 0 and eval_only:
        break

    # * forward backward update
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            X, Y = get_batch(loader)
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps
            scaler.scale(loss).backward()

    # * clip gradients
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    # * optimizer step
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    # * timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5:
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt * 1000:.2f}ms, mfu {running_mfu * 100:.2f}%")

    iter_num += 1
    local_iter_num += 1

    # * termination
    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()