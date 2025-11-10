"""
Sample from a trained model
"""

import os
import pickle
from contextlib import nullcontext
import torch
from trainer_model import Config, Module
from nano import Nano
from tokenizer import WordEncodeResult

init_from = 'resume'
out_dir = '.local/output'
max_new_tokens = 4096
temperature = 0.8
top_k = 200
seed = 1238
device = 'cuda'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compilation = True

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.fp32_precision = 'tf32'
torch.backends.cudnn.conv.fp32_precision = 'tf32'
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# Initialize tokenizer
nano = Nano()

# model
if init_from == 'resume':
    # init from a model saved in a specific directory
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = Config(**checkpoint['model_args'])
    model = Module(gptconf)
    state_dict = checkpoint['model']

    # Remove '_orig_mod.' prefix if present (from compiled models)
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

    model.load_state_dict(state_dict)

model.eval()
model.to(device)

if compilation:
    model = torch.compile(model)  # requires PyTorch 2.0 (optional)


# Encoding and decoding functions using Nano tokenizer
def encode(s):
    """Encode string to token list using Nano tokenizer"""
    result = nano.encode(s)
    parsed = WordEncodeResult(result)
    return parsed.to_token_list()


def decode(tokens):
    """Decode token list to string using Nano tokenizer"""
    decoded_tokens = []
    for token in tokens:
        decoded_tokens.append(nano.decode(token))
    return ''.join(decoded_tokens)


# encode the beginning of the prompt
start = "package"

start_ids = encode(start)
x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]

# run generation
with torch.no_grad():
    with ctx:
        y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
        print(y[0].tolist())
        print(decode(y[0].tolist()))
