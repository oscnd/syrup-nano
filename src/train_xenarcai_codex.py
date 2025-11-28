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
from contextlib import nullcontext
from functools import partial
import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel
from torch.distributed import init_process_group, destroy_process_group
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.distributed.fsdp import FullStateDictConfig, StateDictType
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from trainer_model import Config, Module, Block
from loader import Loader

out_dir = '.local/output'
eval_interval = 16
log_interval = 16
eval_only = False
always_save_checkpoint = True
init_from = 'scratch'
debug = False

wandb_log = True
wandb_project = 'nano'
wandb_run_name = 'nano-run'

dataset_names = ['XenArcAI/CodeX-7M-Non-Thinking']
cache_dir = '.local/cache'
gradient_accumulation_steps = 42
batch_size = 28
block_size = 20480

n_layer = 16
n_head = 16
n_embd = 512
dropout = 0.1
bias = False

learning_rate = 1e-3
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.99
grad_clip = 1.0

decay_lr = True
warmup_iters = 100
min_lr = 1e-4

backend = 'nccl'

device = 'cuda'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compilation = True

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
    assert batch_size % ddp_world_size == 0
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


def get_batch(split='train'):
    local_batch_size = batch_size // ddp_world_size if ddp else batch_size

    x_list = []
    y_list = []

    for _ in range(local_batch_size):
        tokens = loader.get(split=split)

        if tokens is None:
            loader.seek(0, split=split)
            tokens = loader.get(split=split)
            if tokens is None:
                raise ValueError(f"loader {split} split is empty")

        if len(tokens) < block_size + 1:
            padded = np.zeros(block_size + 1, dtype=np.uint16)
            padded[:len(tokens)] = tokens
            tokens = padded
        elif len(tokens) > block_size + 1:
            tokens = tokens[:block_size + 1]

        x = torch.from_numpy(tokens[:block_size].astype(np.int64))
        y = torch.from_numpy(tokens[1:block_size + 1].astype(np.int64))

        x_list.append(x)
        y_list.append(y)

    x = torch.stack(x_list)
    y = torch.stack(y_list)

    if device_type == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)

    return x, y


if master_process:
    print("initializing loaders...")
loader = Loader(dataset_names, cache_dir=cache_dir)

iter_num = 0
best_val_loss = 1e9

meta_vocab_size = loader.dataset_info['vocab_size']

if master_process:
    print(f"using vocab_size {meta_vocab_size}")

total_train_sequences = loader.num_sequences('train')
sequences_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size
max_iters = total_train_sequences // sequences_per_iter
lr_decay_iters = max_iters

if master_process:
    print(f"total train sequences: {total_train_sequences:,}")
    print(f"sequences per iteration: {sequences_per_iter:,}")
    print(f"calculated iteration num: {max_iters:,}")
    print(f"local batch size per gpu: {batch_size // ddp_world_size if ddp else batch_size}")
    print(f"gradient accumulation steps per gpu: {gradient_accumulation_steps // ddp_world_size if ddp else gradient_accumulation_steps}")

model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout)

if master_process:
    print("initializing a new model from scratch")
model_args['vocab_size'] = meta_vocab_size
config = Config(**model_args)
model = Module(config)

model.to(device)

scaler = torch.amp.GradScaler('cuda', enabled=(dtype == 'float16'))

optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)

if compilation:
    if master_process:
        print("compiling the model...")
    unoptimized_model = model
    model = torch.compile(model)

if ddp:
    if master_process:
        print("wrapping model with fsdp...")

    mixed_precision = MixedPrecision(
        param_dtype=torch_dtype,
        reduce_dtype=torch_dtype,
        buffer_dtype=torch_dtype,
    )

    auto_wrap_policy = partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={Block},
    )

    model = FSDP(
        model,
        mixed_precision=mixed_precision,
        auto_wrap_policy=auto_wrap_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=torch.cuda.current_device(),
        limit_all_gathers=True,
        use_orig_params=True,
    )


@torch.no_grad()
def estimate_loss(last_train_batch=None):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_interval)
        for k in range(eval_interval):
            if split == 'train' and last_train_batch is not None:
                X, Y = last_train_batch
            else:
                X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def get_lr(it):
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


if wandb_log and master_process:
    import wandb

    wandb.init(project=wandb_project, name=wandb_run_name, config=None)

t0 = time.time()
local_iter_num = 0
raw_model = model.module if ddp else model
running_mfu = -1.0
last_train_batch = None

if master_process:
    print("starting training loop...")

while True:
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss(last_train_batch)
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
                if ddp:
                    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
                    with FSDP.state_dict_type(raw_model, StateDictType.FULL_STATE_DICT, save_policy):
                        model_state = raw_model.state_dict()
                else:
                    model_state = raw_model.state_dict()

                checkpoint = {
                    'model': model_state,
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                }
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))

    if iter_num == 0 and eval_only:
        break

    local_gradient_accumulation_steps = gradient_accumulation_steps // ddp_world_size if ddp else gradient_accumulation_steps
    for micro_step in range(local_gradient_accumulation_steps):
        if ddp:
            if micro_step == local_gradient_accumulation_steps - 1:
                with ctx:
                    X, Y = get_batch('train')
                    last_train_batch = (X, Y)
                    logits, loss = model(X, Y)
                    loss = loss / local_gradient_accumulation_steps
                scaler.scale(loss).backward()
            else:
                with model.no_sync():
                    with ctx:
                        X, Y = get_batch('train')
                        last_train_batch = (X, Y)
                        logits, loss = model(X, Y)
                        loss = loss / local_gradient_accumulation_steps
                    scaler.scale(loss).backward()
        else:
            with ctx:
                X, Y = get_batch('train')
                last_train_batch = (X, Y)
                logits, loss = model(X, Y)
                loss = loss / local_gradient_accumulation_steps
            scaler.scale(loss).backward()

    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        lossf = loss.item() * local_gradient_accumulation_steps
        if local_iter_num >= 5:
            mfu = raw_model.estimate_mfu(batch_size * local_gradient_accumulation_steps * ddp_world_size, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt * 1000:.2f}ms, mfu {running_mfu * 100:.2f}%")

    iter_num += 1
    local_iter_num += 1

    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()