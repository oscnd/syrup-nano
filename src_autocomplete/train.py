"""
training script for code autocomplete datasets

to run on a single gpu:
$ python train.py --batch_size=32 --compile=False

to run with ddp on 4 gpus on 1 node:
$ torchrun --standalone --nproc_per_node=4 train.py
"""

import os
import socket
import time
import math
from contextlib import nullcontext
from functools import partial
import numpy as np
import torch
from torch.distributed import init_process_group, destroy_process_group
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.distributed.fsdp import FullStateDictConfig, StateDictType
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from trainer_model import Config, Module, Block
from loader import Loader
from load import load
from nano import Nano

# * environment setup
output_dir = '.local/output'
cache_dir = '.local/cache/autocomplete2'
eval_interval = 4
log_interval = 4
eval_only = False
always_save_checkpoint = True
block_continuation = True

# * logging
debug = False
wandb_log = True
wandb_project = 'nano'
wandb_run_name = 'nano-autocomplete-' + time.strftime("%Y%m%d-%H%M%S")

# * training hyperparameters
gradient_accumulation_steps = 4
batch_size = 10
block_size = 512
n_layer = 16
n_head = 16
n_embd = 512
dropout = 0.1
bias = False
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.99
grad_clip = 1.0
learning_rate_decay = False
learning_rate_warmup_iters = 100
learning_rate_baseline = 3e-4
learning_rate_minimum = 3e-4

backend = 'nccl'
fsdp_sharding_strategy = ShardingStrategy.SHARD_GRAD_OP

device = 'cuda'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compilation = True

nano = Nano()

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
    os.makedirs(output_dir, exist_ok=True)

torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.fp32_precision = 'tf32'
torch.backends.cudnn.conv.fp32_precision = 'tf32'
device_type = 'cuda' if 'cuda' in device else 'cpu'
torch_dtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if (device_type == 'cpu' or ddp) else torch.amp.autocast(device_type=device_type, dtype=torch_dtype)

# * state for block_continuation
batch_remainder = {'train': None, 'val': None}
def get_batch(split='train'):
    local_batch_size = batch_size // ddp_world_size if ddp else batch_size

    x_list = []
    y_list = []

    for _ in range(local_batch_size):
        # * check if we have a remainder from previous call
        if block_continuation and batch_remainder[split] is not None:
            tokens = batch_remainder[split]
            batch_remainder[split] = None
        else:
            tokens = loader.get(split=split)

            if debug:
                decoded = ''.join(nano.decode(token) for token in tokens[:100])
                decoded_lines = decoded.split('\n')
                for line in decoded_lines:
                    print(f"    {line}")

            if tokens is None:
                loader.seek(0, split=split)
                tokens = loader.get(split=split)
                if tokens is None:
                    raise ValueError(f"loader {split} split is empty")

        # * handle tokens based on length
        if len(tokens) < block_size + 1:
            if block_continuation:
                # * get next batch to fill the remainder
                next_tokens = loader.get(split=split)
                if next_tokens is None:
                    loader.seek(0, split=split)
                    next_tokens = loader.get(split=split)

                if next_tokens is not None:
                    # * concatenate current and next tokens
                    combined = np.concatenate([tokens, next_tokens])

                    if len(combined) >= block_size + 1:
                        # * use first block_size+1 tokens, store remainder
                        tokens = combined[:block_size + 1]
                        batch_remainder[split] = combined[block_size + 1:]
                    else:
                        # * still not enough, pad
                        padded = np.zeros(block_size + 1, dtype=np.uint16)
                        padded[:len(combined)] = combined
                        tokens = padded
                else:
                    # * no next tokens available, pad
                    padded = np.zeros(block_size + 1, dtype=np.uint16)
                    padded[:len(tokens)] = tokens
                    tokens = padded
            else:
                # * original behavior: pad
                padded = np.zeros(block_size + 1, dtype=np.uint16)
                padded[:len(tokens)] = tokens
                tokens = padded
        elif len(tokens) > block_size + 1:
            if block_continuation:
                # * store remainder for next call
                batch_remainder[split] = tokens[block_size + 1:]
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

    # Use hardcoded cache directory for code autocomplete
    # The load function uses hardcoded JSONL pattern
    constructor = load(cache_dir=cache_dir)

    # Construct cache if needed (this will use hardcoded JSONL files)
    try:
        loader = Loader(['./download/code/*.jsonl'], cache_dir=cache_dir)
        print("loaded existing cache")
    except Exception as e:
        print(f"cache not found or incomplete, constructing new cache: {e}")
        print("constructing cache from JSONL files...")
        constructor.construct(['./download/code/*.jsonl'])
        loader = Loader(['./download/code/*.jsonl'], cache_dir=cache_dir)
        print("cache construction completed")

iter_num = 0
best_val_loss = 1e9

meta_vocab_size = loader.dataset_info['vocab_size']

if master_process:
    print(f"using vocab_size {meta_vocab_size}")

total_train_sequences = loader.num_sequences('train')
sequences_per_iter = gradient_accumulation_steps * batch_size
max_iters = total_train_sequences // sequences_per_iter

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

# * cast model to target dtype
if dtype != 'float32':
    model = model.to(torch_dtype)

# * initialize a gradscaler
use_grad_scaler = (dtype == 'float16' and not ddp)
scaler = torch.amp.GradScaler('cuda', enabled=use_grad_scaler)

optimizer = model.configure_optimizers(weight_decay, learning_rate_baseline, (beta1, beta2), device_type)

if compilation:
    if master_process:
        print("compiling the model...")
    unoptimized_model = model
    model = torch.compile(model)

if ddp:
    if master_process:
        print("wrapping model with fsdp...")

    auto_wrap_policy = partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={Block},
    )

    model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        sharding_strategy=fsdp_sharding_strategy,
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


def get_learning_rate(it):
    if it < learning_rate_warmup_iters:
        return learning_rate_baseline * (it + 1) / (learning_rate_warmup_iters + 1)
    if it > max_iters:
        return learning_rate_minimum
    decay_ratio = (it - learning_rate_warmup_iters) / (max_iters - learning_rate_warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return learning_rate_minimum + coeff * (learning_rate_baseline - learning_rate_minimum)


if wandb_log and master_process:
    import wandb

    config_dict = {
        'output_dir': output_dir,
        'eval_interval': eval_interval,
        'log_interval': log_interval,
        'always_save_checkpoint': always_save_checkpoint,
        'gradient_accumulation_steps': gradient_accumulation_steps,
        'batch_size': batch_size,
        'block_size': block_size,
        'n_layer': n_layer,
        'n_head': n_head,
        'n_embd': n_embd,
        'dropout': dropout,
        'bias': bias,
        'weight_decay': weight_decay,
        'beta1': beta1,
        'beta2': beta2,
        'grad_clip': grad_clip,
        'learning_rate_decay': learning_rate_decay,
        'learning_rate_warmup_iters': learning_rate_warmup_iters,
        'learning_rate_baseline': learning_rate_baseline,
        'learning_rate_minimum': learning_rate_minimum,
        'backend': backend,
        'fsdp_sharding_strategy': str(fsdp_sharding_strategy),
        'device': device,
        'dtype': dtype,
        'compilation': compilation,
        'total_train_sequences': total_train_sequences,
        'sequences_per_iter': sequences_per_iter,
        'max_iters': max_iters,
        'hostname': socket.gethostname(),
    }
    wandb.init(project=wandb_project, name=wandb_run_name, config=config_dict)

t0 = time.time()
local_iter_num = 0
raw_model = model.module if ddp else model
running_flops = -1.0
last_train_batch = None

if master_process:
    print("starting training loop...")

while True:
    lr = get_learning_rate(iter_num) if learning_rate_decay else learning_rate_baseline
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

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
                loss.backward()
            else:
                with model.no_sync():
                    with ctx:
                        X, Y = get_batch('train')
                        last_train_batch = (X, Y)
                        logits, loss = model(X, Y)
                        loss = loss / local_gradient_accumulation_steps
                    loss.backward()
        else:
            with ctx:
                X, Y = get_batch('train')
                last_train_batch = (X, Y)
                logits, loss = model(X, Y)
                loss = loss / local_gradient_accumulation_steps
            if use_grad_scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

    if grad_clip != 0.0:
        if use_grad_scaler:
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    if use_grad_scaler:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()

    optimizer.zero_grad(set_to_none=True)

    t1 = time.time()
    dt = t1 - t0
    t0 = t1

    if iter_num % log_interval == 0 and master_process:
        lossf = loss.item() * local_gradient_accumulation_steps
        if local_iter_num >= 5:
            flops_per_second = raw_model.estimate_flops(batch_size * gradient_accumulation_steps, dt)
            running_flops = flops_per_second if running_flops == -1.0 else 0.9 * running_flops + 0.1 * flops_per_second
        tflops = running_flops / 1e12 if running_flops > 0 else 0
        eta = (max_iters - iter_num) * (dt)
        print(f"iter {iter_num}: loss {lossf:.4f}, tflops {tflops:.2f}, duration {dt:.2f}s (eta {eta//3600:.0f}h {(eta%3600)//60:.0f}m)")

    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss(last_train_batch)
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "tflops": running_flops / 1e12 if running_flops > 0 else 0,
                "learning_rate": lr,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "duration": dt,
            })
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                if ddp:
                    # * save fsdp model state
                    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
                    with FSDP.state_dict_type(raw_model, StateDictType.FULL_STATE_DICT, save_policy):
                        model_state = raw_model.state_dict()

                    # * only rank 0 saves the checkpoint
                    if master_process:
                        # * remove '_orig_mod.' prefix
                        unwanted_prefix = '_orig_mod.'
                        cleaned_state = {}
                        for k, v in model_state.items():
                            if k.startswith(unwanted_prefix):
                                cleaned_state[k[len(unwanted_prefix):]] = v
                            else:
                                cleaned_state[k] = v

                        checkpoint = {
                            'model': cleaned_state,
                            'optimizer': optimizer.state_dict(),
                            'model_args': model_args,
                            'iter_num': iter_num,
                            'best_val_loss': best_val_loss,
                        }
                        if debug:
                            print(f"Saving checkpoint at iteration {iter_num}")
                        torch.save(checkpoint, os.path.join(output_dir, 'ckpt.pt'))
                else:
                    model_state = raw_model.state_dict()

                    # * remove '_orig_mod.' prefix
                    unwanted_prefix = '_orig_mod.'
                    cleaned_state = {}
                    for k, v in model_state.items():
                        if k.startswith(unwanted_prefix):
                            cleaned_state[k[len(unwanted_prefix):]] = v
                        else:
                            cleaned_state[k] = v

                    checkpoint = {
                        'model': cleaned_state,
                        'optimizer': optimizer.state_dict(),
                        'model_args': model_args,
                        'iter_num': iter_num,
                        'best_val_loss': best_val_loss,
                    }
                    print(f"saving checkpoint at iteration {iter_num}")
                    torch.save(checkpoint, os.path.join(output_dir, 'ckpt.pt'))

    iter_num += 1
    local_iter_num += 1

    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()