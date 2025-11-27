"""
Main entrypoint for loading datasets with automatic cache construction
"""

from loader import Loader, DatasetIncompleteError
from loader_constructor import LoaderConstructor
from loader_process_xenarcai_codex import XenarcaiCodexProcessor
from loader_process_nemotron_sft import NemotronSftProcessor


def load(dataset_names: list[str], cache_dir: str = '.local/cache') -> Loader:
    """
    Load datasets with automatic cache construction if needed

    Args:
        dataset_names: List of dataset names to load
        cache_dir: Directory for cached data

    Returns:
        Loader instance ready to use
    """
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

        # * create constructor and register processors
        constructor = LoaderConstructor(cache_dir=cache_dir)
        constructor.register(XenarcaiCodexProcessor)
        constructor.register(NemotronSftProcessor)

        # * construct cache
        constructor.construct(dataset_names)

        # * load the newly constructed cache
        print("\nloading newly constructed cache...")
        loader = Loader(dataset_names, cache_dir=cache_dir)
        print("cache loaded successfully!")
        return loader


def create_loader(dataset_names: list[str], cache_dir: str = '.local/cache') -> Loader:
    """
    Convenience function - alias for load()

    Args:
        dataset_names: List of dataset names to load
        cache_dir: Directory for cached data

    Returns:
        Loader instance ready to use
    """
    return load(dataset_names, cache_dir=cache_dir)


if __name__ == "__main__":
    # * test with xenarcai codex datasets
    dataset_names = [
        "XenArcAI/CodeX-7M-Non-Thinking",
    ]

    loader = load(dataset_names)

    print(f"\nloader ready: {len(loader):,} total tokens")
    print(f"  train sequences: {loader.num_sequences('train'):,}")
    print(f"  val sequences: {loader.num_sequences('val'):,}")

    print("\ntesting first 3 train sequences:")
    for i in range(3):
        tokens = loader.get(split='train')
        if tokens is not None:
            print(f"\nsequence {i + 1}:")
            print(f"  token count: {len(tokens):,}")
            print(f"  first 50 tokens: {tokens[:50]}")
        else:
            print(f"\nsequence {i + 1}: no more data")
            break

    print("\ntesting first 3 val sequences:")
    for i in range(3):
        tokens = loader.get(split='val')
        if tokens is not None:
            print(f"\nsequence {i + 1}:")
            print(f"  token count: {len(tokens):,}")
            print(f"  first 50 tokens: {tokens[:50]}")
        else:
            print(f"\nsequence {i + 1}: no more data")
            break
