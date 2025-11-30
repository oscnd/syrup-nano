"""
sample from a trained model for code autocomplete
"""

import os
import argparse
from contextlib import nullcontext
import torch
from trainer_model import Config, Module
from nano import Nano
from word import WordEncodeResult

# configuration
init_from = 'resume'
out_dir = '.local/output'
num_samples = 1
temperature = 0.8
top_k = 200
seed = 1337
device = 'cpu' if torch.cuda.is_available() else 'cpu'
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


def main():
    parser = argparse.ArgumentParser(description='Generate code autocomplete samples')
    parser.add_argument('--path', type=str, required=True, help='File path for code completion context (e.g., main.go)')
    parser.add_argument('--content', type=str, required=True, help='Code content for completion context (e.g., "package main")')
    parser.add_argument('--predict', type=int, default=16, help='Number of next tokens to predict (default: 16)')
    parser.add_argument('--temperature', type=float, default=0.8, help='Sampling temperature (default: 0.8)')
    parser.add_argument('--top_k', type=int, default=200, help='Top-k sampling (default: 200)')
    parser.add_argument('--out_dir', type=str, default='.local/output', help='Model checkpoint directory (default: .local/output)')

    args = parser.parse_args()

    print(f"Using path: {args.path}")
    print(f"Using content: {args.content}")
    print()

    # Build token sequence following the loader_constructor.py format:
    # [900] + encode(path) + [901] + encode(content) + predict_next_tokens
    tokens = []

    # Add path start marker (900)
    tokens.append(900)

    # Add encoded content (tokenize the content string)
    content_tokens = encode(args.content)
    tokens.extend(content_tokens)

    print(f"Input sequence length: {len(tokens)} tokens")
    print(f"Predicting {args.predict} next tokens...")
    print()

    # convert to tensor
    x = torch.tensor(tokens, dtype=torch.long, device=device)[None, ...]

    # run generation
    with torch.no_grad():
        with ctx:
            y = model.generate(x, args.predict, temperature=args.temperature, top_k=args.top_k)

            # Get only the newly generated tokens (exclude the input)
            generated_tokens = y[0][len(tokens):].tolist()

            # Decode the generated tokens
            generated_text = decode(generated_tokens)

            print("=" * 80)
            print("PREDICTION RESULTS:")
            print("=" * 80)
            print(f"Generated {len(generated_tokens)} tokens: {generated_text}")
            print()
            print("Generated tokens (numeric):")
            print(generated_tokens)
            print("=" * 80)


if __name__ == "__main__":
    # model loading - only when script is run directly
    print(f"loading checkpoint from {out_dir}")
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')

    if not os.path.exists(ckpt_path):
        print(f"Error: No checkpoint found at {ckpt_path}")
        print("Please train a model first using train.py")
        exit(1)

    checkpoint = torch.load(ckpt_path, map_location=device)
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

    print("model ready for inference")
    print()

    main()