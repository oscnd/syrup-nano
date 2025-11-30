"""
Main entrypoint for loading JSONL code autocomplete datasets with automatic cache construction
"""

import argparse
from loader import Loader, DatasetIncompleteError
from loader_constructor import LoaderConstructor
from loader_constructor import Processor
from nano import Nano


def load(cache_dir: str = '.local/cache/gocode') -> Loader:
    """
    Load JSONL datasets with automatic cache construction if needed

    Args:
        cache_dir: Directory for cached data

    Returns:
        Loader instance ready to use
    """
    dataset_names = ['./download/code/*.jsonl']

    try:
        # * attempt to load existing cache
        print(f"attempting to load cache for datasets: {dataset_names}")
        loader = Loader(dataset_names, cache_dir=cache_dir)
        print("cache loaded successfully!")
        return loader

    except DatasetIncompleteError as e:
        # * cache is incomplete or missing, construct it
        print(f"\ncache incomplete: {e}")
        print("constructing cache...")

        # * create constructor and register processor
        constructor = LoaderConstructor(cache_dir=cache_dir)
        constructor.register(Processor)

        # * construct cache
        constructor.construct(dataset_names)

        # * load the newly constructed cache
        print("\nloading newly constructed cache...")
        loader = Loader(dataset_names, cache_dir=cache_dir)
        print("cache loaded successfully!")
        return loader


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Load JSONL code autocomplete datasets with cache construction')
    parser.add_argument(
        '--cache-dir',
        type=str,
        default='.local/cache/autocomplete',
        help='Directory for cached data (default: .local/cache/autocomplete)'
    )

    args = parser.parse_args()

    loader = load(cache_dir=args.cache_dir)
    nano = Nano()

    print(f"\nloader ready: {len(loader):,} total tokens")
    print(f"  train sequences: {loader.num_sequences('train'):,}")
    print(f"  val sequences: {loader.num_sequences('val'):,}")

    print("\ntesting first 3 train sequences:")
    for i in range(3):
        tokens = loader.get(split='train')
        if tokens is not None:
            decoded = ''.join(nano.decode(token) for token in tokens[:100])
            decoded_lines = decoded.split('\n')

            print(f"\nsequence {i + 1}:")
            print(f"\nsequence {i + 1}:")
            print(f"  token count: {len(tokens):,}")
            print(f"  first 100 tokens: {tokens[:100]}")
            print(f"  decoded 100 tokens:")
            for line in decoded_lines:
                print(f"    {line}")
        else:
            print(f"\nsequence {i + 1}: no more data")
            break
