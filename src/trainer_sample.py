"""
sample from a trained model
"""

import os
from contextlib import nullcontext
import torch
from trainer_model import Config, Module
from nano import Nano
from word import WordEncodeResult

# configuration
init_from = 'resume'
out_dir = '.local/output'
start = "#sectionInstructionStart##sectionInstructionEnd##sectionInputStart#Write hello world in python.#sectionInputEnd#"
num_samples = 1
max_new_tokens = 32
temperature = 0.8
top_k = 200
seed = 1337
device = 'cpu'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compilation = True

# setup
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.fp32_precision = 'tf32'
torch.backends.cudnn.conv.fp32_precision = 'tf32'
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# initialize tokenizer
nano = Nano()

# model
if init_from == 'resume':
    # init from a model saved in a specific directory
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)

    print(f"loading checkpoint from {ckpt_path}")
    print(f"model args from checkpoint: {checkpoint['model_args']}")

    gptconf = Config(**checkpoint['model_args'])
    model = Module(gptconf)
    state_dict = checkpoint['model']

    # handle fsdp flattened/empty tensors
    vocab_size = checkpoint['model_args']['vocab_size']
    n_embd = checkpoint['model_args']['n_embd']
    block_size = checkpoint['model_args']['block_size']

    try:
        model.load_state_dict(state_dict, strict=False)
        print("model loaded successfully")
    except Exception as e:
        print(f"error loading model: {e}")
        raise

model.eval()
model.to(device)

# cast model to target dtype
if dtype != 'float32':
    model = model.to(ptdtype)

if compilation:
    print("compiling the model...")
    model = torch.compile(model)


# encoding and decoding functions using nano tokenizer
def encode(s):
    """encode string to token list using nano tokenizer"""
    result = nano.encode(s)
    parsed = WordEncodeResult(result)
    return parsed.to_token_list()


def decode(tokens):
    """decode token list to string using nano tokenizer"""
    decoded_tokens = []
    for token in tokens:
        decoded_tokens.append(nano.decode(token))
    return ''.join(decoded_tokens)


# encode the beginning of the prompt
if start.startswith('file:'):
    with open(start[5:], 'r', encoding='utf-8') as f:
        start = f.read()

print(f"encoding prompt: {start[:50]}...")
start_ids = encode(start)

# convert to tensor
x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]

# run generation
print(f"generating {num_samples} sample(s)...")
with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)

            # decode using nano tokenizer
            decoded = decode(y[0].tolist())

            print('=' * 80)
            print(f"sample {k+1}:")
            print('=' * 80)
            print(decoded)
            print('=' * 80)