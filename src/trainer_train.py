"""
Training script

To run on a single GPU:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node:
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

# * configuration
out_dir = '.local/output'
eval_interval = 500
log_interval = 100
eval_iters = 200
eval_only = False
always_save_checkpoint = False
init_from = 'scratch'
debug = False

# * wandb logging
wandb_log = False
wandb_project = 'nano'
wandb_run_name = 'nano-run'

# * data
train_split = 0.9
gradient_accumulation_steps = 2
batch_size = 32
block_size = 256

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
def get_batch(split, data_iter_pos):
    if split == 'train':
        data = np.memmap(os.path.join(out_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(out_dir, 'val.bin'), dtype=np.uint16, mode='r')

    # * wrap around if we exceed data length
    if data_iter_pos + batch_size * (block_size + 1) > len(data):
        data_iter_pos = 0

    # * sequential batching
    x_list = []
    y_list = []
    for i in range(batch_size):
        offset = data_iter_pos + i * (block_size + 1)
        x_list.append(torch.from_numpy((data[offset:offset + block_size]).astype(np.int64)))
        y_list.append(torch.from_numpy((data[offset + 1:offset + 1 + block_size]).astype(np.int64)))

    x = torch.stack(x_list)
    y = torch.stack(y_list)

    if device_type == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)

    # * update position for next call
    new_pos = data_iter_pos + batch_size * (block_size + 1)
    if new_pos >= len(data):
        new_pos = 0

    return x, y, new_pos


# * init these up here
iter_num = 0
best_val_loss = 1e9
train_data_pos = 0
val_data_pos = 0

# * get vocab_size from dataset
meta_path = os.path.join(out_dir, 'meta.pkl')
with open(meta_path, 'rb') as f:
    meta = pickle.load(f)
meta_vocab_size = meta['vocab_size']
print(f"found vocab_size {meta_vocab_size} (inside {meta_path})")

# * calculate max_iters based on dataset size
train_data = np.memmap(os.path.join(out_dir, 'train.bin'), dtype=np.uint16, mode='r')
total_train_tokens = len(train_data)
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
max_iters = total_train_tokens // tokens_per_iter
lr_decay_iters = max_iters

print(f"total train tokens: {total_train_tokens:,}")
print(f"tokens per iteration: {tokens_per_iter:,}")
print(f"calculated max_iters: {max_iters:,}")

# * model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout)

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
    print("compiling the model...")
    unoptimized_model = model
    model = torch.compile(model)

# * wrap model into ddp container
if ddp:
    model = DistributedDataParallel(model, device_ids=[ddp_local_rank])


# * estimate loss
@torch.no_grad()
def estimate_loss():
    global val_data_pos
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        temp_pos = 0 if split == 'train' else val_data_pos
        for k in range(eval_iters):
            X, Y, temp_pos = get_batch(split, temp_pos)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
        if split == 'val':
            val_data_pos = temp_pos
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

while True:
    # * set learning rate
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # * evaluate and save checkpoints
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
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
                    'train_data_pos': train_data_pos,
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
            X, Y, train_data_pos = get_batch('train', train_data_pos)
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