"""
Data preparation script for tokenizing raw text data into tokens
"""

import os
import pickle
import numpy as np
from nano import Nano
from word import WordEncodeResult
from loader import create_loader

# * configuration
OUT_DIR = '.local/output'
TRAIN_SPLIT = 0.9
CHUNK_SIZE = 1_048_576
DEBUG = False

def prepare_data_chunked():
    """
    Prepare training data in a memory-efficient way by processing in chunks.
    Writes data incrementally to disk instead of loading everything into memory.
    """
    # * construct path
    train_file = os.path.join(OUT_DIR, 'train.bin')
    val_file = os.path.join(OUT_DIR, 'val.bin')
    meta_file = os.path.join(OUT_DIR, 'meta.pkl')
    temp_file = os.path.join(OUT_DIR, 'temp.bin')

    # * check if data already exists
    if os.path.exists(train_file) and os.path.exists(val_file) and os.path.exists(meta_file):
        print(f"Data already prepared in {OUT_DIR}")
        return OUT_DIR

    print(f"preparing data from loader (chunked mode)...")
    os.makedirs(OUT_DIR, exist_ok=True)

    # * create loader
    loader = create_loader()

    if len(loader) == 0:
        raise ValueError("no data found in loader")

    # * initialize nano tokenizer
    nano = Nano()

    # * process data in chunks
    processed_count = 0
    total_tokens = 0
    max_token_value = 0

    print(f"processing {len(loader)} entries in chunks of {CHUNK_SIZE}...")

    # * open temporary file for writing tokens incrementally
    with open(temp_file, 'wb') as temp_f:
        chunk_tokens = []

        while True:
            result = loader.get()
            if result is None:
                break

            metadata, content = result

            if processed_count % 1000 == 0:
                print(f"  Processed {processed_count}/{len(loader)} entries {total_tokens:,} tokens...")

            try:
                if content:
                    # * encode content
                    result = nano.encode(content)
                    parsed = WordEncodeResult(result)
                    tokens = parsed.to_token_list()

                    # * add to current chunk
                    chunk_tokens.extend(tokens)
                    total_tokens += len(tokens)
                    processed_count += 1

                    # * track max token value for vocab size
                    if tokens:
                        max_token_value = max(max_token_value, max(tokens))

                    if DEBUG:
                        print(f"  Content: {content[:30]!r}... => {len(tokens)} tokens")
                        print(f"    Tokens: {tokens[:60]}...")

                    # * flush chunk to disk
                    if len(chunk_tokens) >= CHUNK_SIZE:
                        chunk_array = np.array(chunk_tokens, dtype=np.uint16)
                        chunk_array.tofile(temp_f)
                        chunk_tokens = []

            except Exception as e:
                print(f"  Error processing {metadata}: {e}")
                continue

        # * write remaining tokens
        if chunk_tokens:
            chunk_array = np.array(chunk_tokens, dtype=np.uint16)
            chunk_array.tofile(temp_f)
            chunk_tokens = []

    print(f"\ntotal entries processed: {processed_count:,}")
    print(f"total tokens: {total_tokens:,}")

    # * calculate vocab size
    vocab_size = max_token_value + 1
    print(f"vocabulary size: {vocab_size:,}")

    # * split into train and validation
    print("splitting into train and validation sets...")
    split_idx = int(total_tokens * TRAIN_SPLIT)

    print(f"train tokens: {split_idx:,}")
    print(f"validation tokens: {total_tokens - split_idx:,}")

    # * load all tokens
    all_tokens = np.memmap(temp_file, dtype=np.uint16, mode='r', shape=(total_tokens,))

    # * create train file
    print("writing train file...")
    train_tokens = np.memmap(train_file, dtype=np.uint16, mode='w+', shape=(split_idx,))

    # * copy in chunks
    for i in range(0, split_idx, CHUNK_SIZE):
        end_idx = min(i + CHUNK_SIZE, split_idx)
        train_tokens[i:end_idx] = all_tokens[i:end_idx]
        if i % (CHUNK_SIZE * 10) == 0:
            print(f"  written {i:,}/{split_idx:,} train tokens...")

    train_tokens.flush()
    del train_tokens

    # * create validation file
    print("writing validation file...")
    val_size = total_tokens - split_idx
    val_tokens = np.memmap(val_file, dtype=np.uint16, mode='w+', shape=(val_size,))

    for i in range(0, val_size, CHUNK_SIZE):
        end_idx = min(i + CHUNK_SIZE, val_size)
        val_tokens[i:end_idx] = all_tokens[split_idx + i:split_idx + end_idx]
        if i % (CHUNK_SIZE * 10) == 0:
            print(f"  written {i:,}/{val_size:,} validation tokens...")

    val_tokens.flush()
    del val_tokens
    del all_tokens

    # * cleanup
    print("cleaning up temporary files...")
    os.remove(temp_file)

    # * save metadata
    meta = {
        'vocab_size': vocab_size,
        'train_size': split_idx,
        'val_size': val_size,
    }

    with open(meta_file, 'wb') as f:
        pickle.dump(meta, f)

    print(f"data preparation completed and saved to {OUT_DIR}")
    print(f"  train file: {train_file}")
    print(f"  validation file: {val_file}")
    print(f"  metadata file: {meta_file}")

    return OUT_DIR


if __name__ == "__main__":
    prepare_data_chunked()