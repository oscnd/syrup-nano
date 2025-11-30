"""
Loader constructor for building dataset cache with JSONL processor for code autocomplete
"""

import os
import re
import json
import time
import glob
import numpy as np
from abc import ABC, abstractmethod
from typing import Optional
from nano import Nano
from word import WordEncodeResult


class LoaderConstructorProcessor(ABC):
    """Abstract base class for dataset processors"""

    def __init__(self, nano: Nano, section_markers: dict):
        self.nano = nano
        self.section_markers = section_markers

    @abstractmethod
    def check(self, dataset_name: str) -> bool:
        """Check if this processor can handle the given dataset name"""
        pass

    @abstractmethod
    def subsets(self) -> list[str]:
        """Return list of dataset subsets this processor can handle"""
        pass

    @abstractmethod
    def should_filter(self) -> bool:
        """
        Check if filtering is required for this dataset

        Returns:
            True if filtering should be applied, False otherwise
        """
        pass

    @abstractmethod
    def filter(self, row: dict) -> bool:
        """
        Filter function to apply to dataset rows

        Args:
            row: The dataset row

        Returns:
            True if row should be kept, False if it should be filtered out
        """
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


class Processor(LoaderConstructorProcessor):
    """Processor for local JSONL files containing code data"""

    def __init__(self, nano: Nano, section_markers: dict):
        super().__init__(nano, section_markers)
        self.jsonl_files = []
        self.current_file_idx = 0
        self.current_line_idx = 0

    def check(self, dataset_name: str) -> bool:
        """Check if this processor can handle the given dataset name"""
        # Accept any dataset name that contains jsonl or is the default pattern
        return 'jsonl' in dataset_name.lower() or dataset_name == './download/code/*.jsonl'

    def subsets(self) -> list[str]:
        """Return list of dataset subsets this processor can handle"""
        # No subsets for JSONL files
        return []

    def should_filter(self) -> bool:
        """Check if filtering is required for this dataset"""
        # No filtering needed for JSONL files
        return False

    def filter(self, row: dict) -> bool:
        """Filter function to apply to dataset rows"""
        # Pass-through all rows
        return True

    def process_row(self, row: dict, dataset_name: str) -> Optional[np.ndarray]:
        """Process a single JSONL row and return tokens"""
        try:
            # extract path and content from JSONL row
            path = row.get('path', '')
            content = row.get('content', '')

            # remove import blocks
            content = re.sub(r'import\s*\([^)]*\)', '', content)

            # merge consecutive newlines
            content = re.sub(r'\n{3,}', '\n\n', content)

            if not content:
                return None

            tokens = []

            # add sequence marker
            tokens.append(900)

            # Add encoded content
            content_tokens = self._encode_text(content)
            tokens.extend(content_tokens)

            # add sequence marker
            tokens.append(902)

            return np.array(tokens, dtype=np.uint16)

        except Exception as e:
            print(f"Error processing row: {e}")
            return None

    def get_jsonl_files(self) -> list[str]:
        """Get list of JSONL files to process"""
        if not self.jsonl_files:
            # Use absolute path to the project root to ensure files are found from any directory
            # First try to find the project root by looking for go.mod or going up from src_autocomplete
            current_dir = os.getcwd()
            if current_dir.endswith('src_autocomplete'):
                # We're in src_autocomplete, go up one level to project root
                project_root = os.path.dirname(current_dir)
            else:
                # Try to find project root by looking for go.mod or other markers
                project_root = current_dir
                while project_root != '/':
                    if os.path.exists(os.path.join(project_root, 'go.mod')):
                        break
                    if os.path.basename(project_root) == 'nano':
                        break
                    project_root = os.path.dirname(project_root)
                else:
                    project_root = current_dir

            pattern = os.path.join(project_root, 'download', 'code', '*.jsonl')
            self.jsonl_files = glob.glob(pattern)
            print(f"Found {len(self.jsonl_files)} JSONL files matching pattern: {pattern}")
        return self.jsonl_files

    def iterate_rows(self):
        """Iterate over all rows in all JSONL files"""
        for file_path in self.get_jsonl_files():
            print(f"Processing file: {file_path}")
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            row = json.loads(line)
                            yield row, f"{file_path}:{line_num}"
                        except json.JSONDecodeError as e:
                            print(f"Error parsing JSON at {file_path}:{line_num}: {e}")
                            continue
            except Exception as e:
                print(f"Error reading file {file_path}: {e}")
                continue


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
            if processor.check(dataset_name):
                return processor
        return None

    def _save_progress(self, dataset_metadata, full_sequences, total_tokens, max_token_value):
        """Save progress to index.json"""
        with open(self.index_json_file, 'w') as f:
            json.dump({
                'datasets': dataset_metadata,
                'full_sequences': full_sequences,
                'total_tokens': total_tokens,
                'vocab_size': max_token_value + 1,
                'train_split': self.train_split
            }, f, indent=2)

    def _load_progress(self):
        """Load progress from index.json if exists"""
        if os.path.exists(self.index_json_file):
            with open(self.index_json_file, 'r') as f:
                progress = json.load(f)

                # * check if all datasets are complete
                all_complete = True
                for ds in progress.get('datasets', []):
                    if ds.get('processed_sequences', 0) < ds.get('total_sequences', 0):
                        all_complete = False
                        break

                # * only return progress if not all complete
                if not all_complete:
                    return progress
        return None

    def construct(self, dataset_names: list[str]):
        """
        Encode JSONL datasets and flush to binary files

        Args:
            dataset_names: List of dataset names to process (will use hardcoded JSONL pattern)
        """
        os.makedirs(self.cache_dir, exist_ok=True)

        # * Use hardcoded JSONL pattern regardless of input
        dataset_pattern = './download/code/*.jsonl'
        processor = self._find_processor(dataset_pattern)

        if processor is None:
            raise ValueError(f"No processor found for JSONL pattern: {dataset_pattern}")

        # * check for resume
        progress = self._load_progress()

        if progress:
            print(f"resuming from previous progress...")
            total_tokens = progress['total_tokens']
            max_token_value = progress['vocab_size'] - 1
            dataset_metadata = progress['datasets']

            # * calculate full sequences from metadata
            full_sequences = sum(ds['processed_sequences'] for ds in dataset_metadata)

            # * find incomplete dataset (if any) - must complete first to maintain sequential order
            incomplete_dataset_idx = None
            for idx, ds in enumerate(dataset_metadata):
                if ds['processed_sequences'] < ds['total_sequences']:
                    incomplete_dataset_idx = idx
                    print(f"found incomplete dataset: {ds['name']}")
                    print(f"  processed: {ds['processed_sequences']:,}/{ds['total_sequences']:,}")
                    print(f"  must complete this dataset first to maintain sequential ordering in data.bin")
                    break

            # * determine which dataset to resume from
            if incomplete_dataset_idx is not None:
                resume_dataset_idx = incomplete_dataset_idx
            else:
                resume_dataset_idx = len(dataset_metadata)

            if resume_dataset_idx >= len(dataset_names):
                print("all datasets already processed")
                self._save_progress(dataset_metadata, full_sequences, total_tokens, max_token_value)
                return
        else:
            print("starting fresh...")
            total_tokens = 0
            max_token_value = 0
            full_sequences = 0
            dataset_metadata = []
            resume_dataset_idx = 0

        print("writing tokens...")

        # * open files in append mode if resuming, write mode if starting fresh
        data_mode = 'ab' if progress else 'wb'
        index_mode = 'ab' if progress else 'wb'

        with open(self.data_file, data_mode) as data_f, open(self.index_file, index_mode) as index_f:
            chunk_size = 16_777_216
            chunk_tokens = []
            index_buffer = []

            # * Use hardcoded dataset pattern
            dataset_full_name = './download/code/*.jsonl'

            # * get JSONL files from processor
            jsonl_files = processor.get_jsonl_files()
            dataset_total_sequences = 0

            # * count total sequences first
            print("counting total sequences...")
            for file_path in jsonl_files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                try:
                                    row = json.loads(line)
                                    if row.get('content'):  # Only count rows with content
                                        dataset_total_sequences += 1
                                except json.JSONDecodeError:
                                    continue
                except Exception as e:
                    print(f"Error counting sequences in {file_path}: {e}")
                    continue

            print(f"dataset total sequences: {dataset_total_sequences:,}")

            # * determine start sequence index for this dataset
            if len(dataset_metadata) > 0:
                dataset_start_seq = dataset_metadata[0]['start_sequence_idx']
            else:
                dataset_start_seq = full_sequences

            processed_sequences = 0

            # * if resuming current dataset, skip already processed sequences
            skip_count = 0
            if resume_dataset_idx == 0 and len(dataset_metadata) > 0:
                skip_count = dataset_metadata[0].get('processed_sequences', 0)
                if skip_count > 0:
                    print(f"resuming from sequence {skip_count:,}...")
                    print(f"skipping first {skip_count:,} sequences...")

            # * time performance
            t0 = time.perf_counter()
            ps0 = processed_sequences
            pt0 = total_tokens

            # * iterate through all JSONL files and process rows
            for row, row_id in processor.iterate_rows():
                # * skip already processed sequences
                if processed_sequences < skip_count:
                    processed_sequences += 1
                    continue

                try:
                    tokens = processor.process_row(row, dataset_full_name)

                    if tokens is None:
                        continue

                    # * add tokens to chunk buffer
                    chunk_tokens.extend(tokens.tolist())

                    # * add index to index buffer (points to where this sequence starts)
                    index_buffer.append(total_tokens)

                    # * update counters
                    total_tokens += len(tokens)
                    full_sequences += 1
                    processed_sequences += 1

                    # * track max token value
                    if len(tokens) > 0:
                        max_token_value = max(max_token_value, int(tokens.max()))

                    # * flush chunk to disk and save progress
                    if len(chunk_tokens) >= chunk_size:
                        # * write data chunk
                        chunk_array = np.array(chunk_tokens, dtype=np.uint16)
                        chunk_array.tofile(data_f)
                        chunk_tokens = []

                        # * write index buffer
                        if index_buffer:
                            index_array = np.array(index_buffer, dtype=np.uint64)
                            index_array.tofile(index_f)
                            index_buffer = []

                        # * update current dataset metadata
                        current_ds_meta = {
                            'name': dataset_full_name,
                            'start_sequence_idx': dataset_start_seq,
                            'total_sequences': dataset_total_sequences,
                            'processed_sequences': processed_sequences
                        }

                        if len(dataset_metadata) == 0:
                            dataset_metadata.append(current_ds_meta)
                        else:
                            dataset_metadata[0] = current_ds_meta

                        # * save progress to disk
                        self._save_progress(dataset_metadata, full_sequences, total_tokens, max_token_value)

                        # * print progress
                        delta_t = time.perf_counter() - t0
                        delta_sequences = (processed_sequences - ps0) / delta_t
                        delta_tokens = (total_tokens - pt0) / delta_t
                        t0 = time.perf_counter()
                        ps0 = processed_sequences
                        pt0 = total_tokens
                        print(
                            f"  processed {processed_sequences:,}/{dataset_total_sequences:,} ({delta_sequences :,.2f}/s) sequences, {total_tokens} ({delta_tokens :,.2f}/s) tokens"
                        )

                except Exception as e:
                    print(f"error processing row {processed_sequences} from {row_id}: {e}")
                    continue

            # * after finishing dataset, flush any remaining data and save final state
            # * write remaining tokens for this dataset
            if chunk_tokens:
                chunk_array = np.array(chunk_tokens, dtype=np.uint16)
                chunk_array.tofile(data_f)
                chunk_tokens = []

            # * write remaining indices for this dataset
            if index_buffer:
                index_array = np.array(index_buffer, dtype=np.uint64)
                index_array.tofile(index_f)
                index_buffer = []

            # * save dataset metadata with final count
            current_ds_meta = {
                'name': dataset_full_name,
                'start_sequence_idx': dataset_start_seq,
                'total_sequences': dataset_total_sequences,
                'processed_sequences': processed_sequences
            }

            if len(dataset_metadata) == 0:
                dataset_metadata.append(current_ds_meta)
            else:
                dataset_metadata[0] = current_ds_meta

            # * save progress after completing dataset
            self._save_progress(dataset_metadata, full_sequences, total_tokens, max_token_value)

            print(f"completed {dataset_full_name}: {processed_sequences:,}/{dataset_total_sequences:,} sequences")

        # * final writes at end of all datasets (should be empty if datasets completed properly)
        with open(self.data_file, 'ab') as data_f, open(self.index_file, 'ab') as index_f:
            # * write any remaining tokens
            if chunk_tokens:
                print("  writing final token chunk...")
                chunk_array = np.array(chunk_tokens, dtype=np.uint16)
                chunk_array.tofile(data_f)

            # * write any remaining indices
            if index_buffer:
                print("  writing final index buffer...")
                index_array = np.array(index_buffer, dtype=np.uint64)
                index_array.tofile(index_f)

        print(f"\nfull sequences processed: {full_sequences:,}")
        print(f"total tokens: {total_tokens:,}")

        # * calculate vocab size
        vocab_size = max_token_value + 1
        print(f"vocabulary size: {vocab_size:,}")

        # * save final progress
        self._save_progress(dataset_metadata, full_sequences, total_tokens, max_token_value)

        print(f"cache construction completed and saved to {self.cache_dir}")