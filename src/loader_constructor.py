"""
Loader constructor for building dataset cache with pluggable processors
"""

import os
import json
import numpy as np
from abc import ABC, abstractmethod
from typing import Optional
from datasets import load_dataset
from nano import Nano
from word import WordEncodeResult


class LoaderConstructorProcessor(ABC):
    """Abstract base class for dataset processors"""

    def __init__(self, nano: Nano, section_markers: dict):
        self.nano = nano
        self.section_markers = section_markers

    @abstractmethod
    def can_process(self, dataset_name: str) -> bool:
        """Check if this processor can handle the given dataset name"""
        pass

    @abstractmethod
    def process_row(self, row: dict, dataset_name: str) -> Optional[np.ndarray]:
        """
        Process a single row and return tokens

        Args:
            row: The dataset row
            dataset_name: Full dataset name for context

        Returns:
            numpy array of uint16 tokens or None if row should be skipped
        """
        pass

    @abstractmethod
    def should_filter(self, row: dict) -> bool:
        """
        Check if a row should be filtered out

        Returns:
            True if row should be kept, False if it should be filtered out
        """
        pass

    def _encode_text(self, text: str) -> list[int]:
        """Encode text to tokens"""
        if not text:
            return []
        text = text.strip()
        if not text:
            return []
        result = self.nano.encode(text)
        parsed = WordEncodeResult(result)
        return parsed.to_token_list()


class LoaderConstructor:
    def __init__(self, cache_dir: str = '.local/cache', train_split: float = 0.9):
        """
        Initialize loader constructor

        Args:
            cache_dir: Directory to store cached data
            train_split: Ratio for train/val split (default: 0.9)
        """
        self.cache_dir = cache_dir
        self.train_split = train_split
        self.nano = Nano()

        # * file paths
        self.data_file = os.path.join(self.cache_dir, 'data.bin')
        self.index_file = os.path.join(self.cache_dir, 'index.bin')
        self.index_json_file = os.path.join(self.cache_dir, 'index.json')

        # * registered processors
        self.processors: list[LoaderConstructorProcessor] = []

        # * section markers
        self.section_markers = {}
        self._initialize_markers()

    def _initialize_markers(self):
        """Initialize and encode section markers"""
        marker_names = [
            '#sectionInstructionStart#',
            '#sectionTemplate1#',
            '#sectionInstructionEnd#',
            '#sectionInputStart#',
            '#sectionInputEnd#',
            '#sectionThinkingStart#',
            '#sectionThinkingEnd#',
            '#sectionOutputStart#',
            '#sectionOutputEnd#',
        ]

        for marker in marker_names:
            result = self.nano.encode(marker)
            parsed = WordEncodeResult(result)
            self.section_markers[marker] = parsed.to_token_list()

    def register(self, processor_class: type[LoaderConstructorProcessor]):
        """Register a processor class"""
        processor = processor_class(self.nano, self.section_markers)
        self.processors.append(processor)
        print(f"registered processor: {processor_class.__name__}")

    def _find_processor(self, dataset_name: str) -> Optional[LoaderConstructorProcessor]:
        """Find a processor that can handle the given dataset"""
        for processor in self.processors:
            if processor.can_process(dataset_name):
                return processor
        return None

    def _save_progress(self, dataset_metadata, total_sequences, total_tokens, max_token_value, complete=False):
        """Save progress to index.json"""
        with open(self.index_json_file, 'w') as f:
            json.dump({
                'datasets': dataset_metadata,
                'total_sequences': total_sequences,
                'total_tokens': total_tokens,
                'vocab_size': max_token_value + 1,
                'train_split': self.train_split,
                'complete': complete
            }, f, indent=2)

    def _load_progress(self):
        """Load progress from index.json if exists"""
        if os.path.exists(self.index_json_file):
            with open(self.index_json_file, 'r') as f:
                progress = json.load(f)
                if not progress.get('complete', False):
                    return progress
        return None

    def construct(self, dataset_names: list[str]):
        """
        Encode datasets and flush to binary files

        Args:
            dataset_names: List of dataset names to process
        """
        os.makedirs(self.cache_dir, exist_ok=True)

        # * validate all datasets have processors
        for dataset_name in dataset_names:
            processor = self._find_processor(dataset_name)
            if processor is None:
                raise ValueError(f"No processor found for dataset: {dataset_name}")

        # * check for resume
        progress = self._load_progress()

        if progress:
            print(f"resuming from previous progress...")
            total_tokens = progress['total_tokens']
            max_token_value = progress['vocab_size'] - 1
            dataset_metadata = progress['datasets']

            # * calculate total sequences from metadata
            total_sequences = sum(ds['num_sequences'] for ds in dataset_metadata)

            # * determine which dataset to resume from
            resume_dataset_idx = len(dataset_metadata)
            if resume_dataset_idx > 0:
                last_ds = dataset_metadata[-1]
                resume_from = last_ds.get('processed_sequence_idx', 0)
                print(f"last dataset: {last_ds['name']}, processed {resume_from} sequences")

            if resume_dataset_idx >= len(dataset_names):
                print("all datasets already processed, marking as complete")
                self._save_progress(dataset_metadata, total_sequences, total_tokens, max_token_value, complete=True)
                return
        else:
            print("starting fresh...")
            total_tokens = 0
            max_token_value = 0
            total_sequences = 0
            dataset_metadata = []
            resume_dataset_idx = 0

        print("writing tokens...")

        # * open files in append mode if resuming, write mode if starting fresh
        data_mode = 'ab' if progress else 'wb'
        index_mode = 'ab' if progress else 'wb'

        with open(self.data_file, data_mode) as data_f, open(self.index_file, index_mode) as index_f:
            chunk_tokens = []
            chunk_size = 1_000_000

            index_buffer = []
            index_buffer_size = 10000

            for dataset_idx, dataset_name in enumerate(dataset_names):
                # * skip already completed datasets
                if dataset_idx < resume_dataset_idx:
                    print(f"skipping completed dataset: {dataset_name}")
                    continue

                # * find processor for this dataset
                processor = self._find_processor(dataset_name)
                if processor is None:
                    print(f"ERROR: no processor for {dataset_name}, skipping")
                    continue

                print(f"processing dataset: {dataset_name}")
                dataset = load_dataset(dataset_name, split='train')

                # * apply filter if processor defines one
                try:
                    dataset = dataset.filter(processor.should_filter)
                except Exception as e:
                    print(f"warning: filter failed for {dataset_name}: {e}")

                dataset_start_seq = total_sequences
                processed_sequences = 0

                # * if resuming current dataset, skip already processed sequences
                skip_count = 0
                if dataset_idx == resume_dataset_idx and progress and len(dataset_metadata) > 0:
                    skip_count = dataset_metadata[-1].get('processed_sequence_idx', 0)
                    if skip_count > 0:
                        print(f"skipping first {skip_count} sequences...")

                for row in dataset:
                    # * skip already processed sequences
                    if processed_sequences < skip_count:
                        processed_sequences += 1
                        continue

                    try:
                        tokens = processor.process_row(row, dataset_name)

                        if tokens is None:
                            continue

                        # * record starting index of this sequence
                        index_buffer.append(total_tokens)

                        # * flush index buffer if needed
                        if len(index_buffer) >= index_buffer_size:
                            index_array = np.array(index_buffer, dtype=np.uint64)
                            index_array.tofile(index_f)
                            index_buffer = []

                        chunk_tokens.extend(tokens.tolist())
                        total_tokens += len(tokens)
                        total_sequences += 1
                        processed_sequences += 1

                        # * track max token value
                        if len(tokens) > 0:
                            max_token_value = max(max_token_value, int(tokens.max()))

                        if processed_sequences % 1000 == 0:
                            print(f"  processed {processed_sequences:,} sequences, {total_tokens:,} tokens...")

                        # * flush chunk to disk and save progress
                        if len(chunk_tokens) >= chunk_size:
                            chunk_array = np.array(chunk_tokens, dtype=np.uint16)
                            chunk_array.tofile(data_f)
                            chunk_tokens = []

                            # * update or add current dataset metadata
                            current_ds_meta = {
                                'name': dataset_name,
                                'start_sequence_idx': dataset_start_seq,
                                'total_sequences': total_sequences - dataset_start_seq,
                                'processed_sequences': processed_sequences
                            }

                            if dataset_idx == len(dataset_metadata):
                                dataset_metadata.append(current_ds_meta)
                            else:
                                dataset_metadata[dataset_idx] = current_ds_meta

                            # * save progress
                            self._save_progress(dataset_metadata, total_sequences, total_tokens, max_token_value,
                                                complete=False)

                    except Exception as e:
                        print(f"error processing row {processed_sequences} from {dataset_name}: {e}")
                        continue

                # * save dataset metadata with final count
                current_ds_meta = {
                    'name': dataset_name,
                    'num_sequences': total_sequences - dataset_start_seq,
                    'start_sequence_idx': dataset_start_seq,
                    'processed_sequence_idx': processed_sequences
                }

                if dataset_idx == len(dataset_metadata):
                    dataset_metadata.append(current_ds_meta)
                else:
                    dataset_metadata[dataset_idx] = current_ds_meta

                print(f"completed {dataset_name}: {processed_sequences:,} sequences")

            # * write remaining tokens
            if chunk_tokens:
                chunk_array = np.array(chunk_tokens, dtype=np.uint16)
                chunk_array.tofile(data_f)

            # * write remaining indices
            if index_buffer:
                index_array = np.array(index_buffer, dtype=np.uint64)
                index_array.tofile(index_f)

        print(f"\ntotal sequences processed: {total_sequences:,}")
        print(f"total tokens: {total_tokens:,}")

        # * calculate vocab size
        vocab_size = max_token_value + 1
        print(f"vocabulary size: {vocab_size:,}")

        # * mark as complete
        self._save_progress(dataset_metadata, total_sequences, total_tokens, max_token_value, complete=True)

        print(f"cache construction completed and saved to {self.cache_dir}")