"""
Main loader class for loading cached dataset with progress management
"""

import os
import json
import numpy as np
from typing import Optional


class DatasetIncompleteError(Exception):
    """Raised when a dataset is incomplete or missing"""
    pass


class Loader:
    def __init__(self, dataset_names: list[str], cache_dir: str = '.local/cache'):
        """
        Initialize loader with specified datasets

        Args:
            dataset_names: List of dataset names to load
            cache_dir: Directory containing cached data
        """
        self.dataset_names = dataset_names
        self.cache_dir = cache_dir
        self.train_split = 0.9

        # * file paths
        self.data_file = os.path.join(self.cache_dir, 'data.bin')
        self.index_file = os.path.join(self.cache_dir, 'index.bin')
        self.index_json_file = os.path.join(self.cache_dir, 'index.json')

        # * memory mapped arrays
        self.data = None
        self.indices = None
        self.dataset_info = None

        # * pre-computed filtered indices for O(1) operations
        self.filtered_train_indices = None
        self.filtered_val_indices = None

        # * current position trackers
        self.current_train_idx = 0
        self.current_val_idx = 0

        # * validate and load cache
        self._validate_cache()
        self._load_cache()

    def _validate_cache(self):
        """Validate that cache exists and contains all required datasets"""
        # * check if cache files exist
        if not os.path.exists(self.index_json_file):
            raise DatasetIncompleteError("Cache not found - no index.json exists")

        # * load and check index.json
        with open(self.index_json_file, 'r') as f:
            info = json.load(f)

        # * check if cache is complete
        if not info.get('complete', False):
            raise DatasetIncompleteError("Cache construction incomplete")

        # * check if all required datasets are present
        cached_datasets = {ds['name'] for ds in info.get('datasets', [])}
        required_datasets = set(self.dataset_names)

        missing = required_datasets - cached_datasets
        if missing:
            raise DatasetIncompleteError(
                f"Missing datasets in cache: {', '.join(missing)}"
            )

        # * check if any cached dataset is incomplete
        for ds in info.get('datasets', []):
            if ds['name'] in required_datasets:
                processed = ds.get('processed_sequence_idx', 0)
                total = ds.get('num_sequences', 0)
                if processed < total:
                    raise DatasetIncompleteError(
                        f"Dataset '{ds['name']}' is incomplete: {processed}/{total} sequences"
                    )

        # * check if binary files exist
        if not os.path.exists(self.data_file):
            raise DatasetIncompleteError("Missing data.bin file")
        if not os.path.exists(self.index_file):
            raise DatasetIncompleteError("Missing index.bin file")

    def _load_cache(self):
        """Load memory-mapped arrays from cached files"""
        # * load memmap
        self.data = np.memmap(self.data_file, dtype=np.uint16, mode='r')
        self.indices = np.memmap(self.index_file, dtype=np.uint64, mode='r')

        # * load index.json
        with open(self.index_json_file, 'r') as f:
            full_dataset_info = json.load(f)

        # * filter dataset_info to only include requested datasets
        filtered_datasets = [
            ds for ds in full_dataset_info['datasets']
            if ds['name'] in self.dataset_names
        ]

        # * build optimized index mapping for filtered datasets only
        # this maps filtered sequence indices to absolute indices in the full cache
        self.filtered_train_indices = []
        self.filtered_val_indices = []

        for ds in filtered_datasets:
            start_seq = ds['start_sequence_idx']
            num_seqs = ds['num_sequences']

            # * calculate train/val split for this dataset
            train_seqs = int(num_seqs * self.train_split)
            val_seqs = num_seqs - train_seqs

            # * add absolute indices for train sequences
            for i in range(train_seqs):
                self.filtered_train_indices.append(start_seq + i)

            # * add absolute indices for val sequences
            for i in range(val_seqs):
                self.filtered_val_indices.append(start_seq + train_seqs + i)

        # * convert to numpy arrays for fast indexing
        self.filtered_train_indices = np.array(self.filtered_train_indices, dtype=np.uint64)
        self.filtered_val_indices = np.array(self.filtered_val_indices, dtype=np.uint64)

        # * update dataset_info to reflect filtered view
        self.dataset_info = {
            'datasets': filtered_datasets,
            'total_sequences': len(self.filtered_train_indices) + len(self.filtered_val_indices),
            'total_tokens': full_dataset_info['total_tokens'],
            'vocab_size': full_dataset_info['vocab_size'],
            'train_split': self.train_split,
            'complete': full_dataset_info['complete']
        }

        print(f"loaded cache: {len(self.data):,} total tokens in cache")
        print(f"active datasets: {len(self.dataset_info['datasets'])}")
        for ds in self.dataset_info['datasets']:
            print(f"  - {ds['name']}: {ds['num_sequences']:,} sequences")
        print(f"filtered view: {len(self.filtered_train_indices):,} train, {len(self.filtered_val_indices):,} val")

        # * reset position trackers
        self.current_train_idx = 0
        self.current_val_idx = 0

    def _get_absolute_sequence_index(self, split: str) -> Optional[int]:
        """
        Get the absolute sequence index for current position using pre-computed filtered indices

        Returns:
            absolute sequence index or None if exhausted
        """
        if split == 'train':
            if self.current_train_idx >= len(self.filtered_train_indices):
                return None
            return int(self.filtered_train_indices[self.current_train_idx])
        else:
            if self.current_val_idx >= len(self.filtered_val_indices):
                return None
            return int(self.filtered_val_indices[self.current_val_idx])

    def get(self, split: str = 'train') -> Optional[np.ndarray]:
        """
        Get next complete sequence from specified split

        Args:
            split: 'train' or 'val'

        Returns:
            array of tokens for one complete example
        """
        if self.data is None or self.indices is None:
            return None

        # * get absolute sequence index using pre-computed filtered indices
        seq_idx = self._get_absolute_sequence_index(split)

        if seq_idx is None:
            return None

        # * get start index of current sequence
        start_idx = int(self.indices[seq_idx])

        # * get end index
        if seq_idx + 1 < len(self.indices):
            end_idx = int(self.indices[seq_idx + 1])
        else:
            end_idx = len(self.data)

        # * extract sequence
        sequence = self.data[start_idx:end_idx]

        # * increment position tracker
        if split == 'train':
            self.current_train_idx += 1
        else:
            self.current_val_idx += 1

        return np.array(sequence, dtype=np.uint16)

    def seek(self, position: int, split: str = 'train'):
        """
        Seek to a specific position in the split

        Args:
            position: the position index to seek to (0-based, relative to filtered split)
            split: 'train' or 'val'
        """
        # * validate position is within bounds
        if split == 'train':
            max_pos = len(self.filtered_train_indices)
            if position < 0 or position >= max_pos:
                raise ValueError(f"Position {position} out of range [0, {max_pos})")
            self.current_train_idx = position
        else:
            max_pos = len(self.filtered_val_indices)
            if position < 0 or position >= max_pos:
                raise ValueError(f"Position {position} out of range [0, {max_pos})")
            self.current_val_idx = position

        print(f"seeked to position {position} in {split} split")

    def num_sequences(self, split: str = 'train') -> int:
        """Return total number of sequences in specified split (filtered view)"""
        if split == 'train':
            return len(self.filtered_train_indices)
        else:
            return len(self.filtered_val_indices)

    def __len__(self):
        """Return total number of tokens"""
        if self.data is not None:
            return len(self.data)
        return 0