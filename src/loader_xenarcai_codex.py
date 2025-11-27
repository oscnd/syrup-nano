"""
loader class for dataset XenArcAI/CodeX-*
stores all data in single files with dynamic train/test split
with resume capability
"""

import os
import re
import json
import numpy as np
from datasets import load_dataset
from typing import Optional, List
from nano import Nano
from word import WordEncodeResult


class Loader:
    def __init__(self):
        self.nano = Nano()
        self.cache_dir = '.local/cache'

        # * train / val split ratio
        self.train_split = 0.9

        # * file paths
        self.data_file = os.path.join(self.cache_dir, 'data.bin')
        self.index_file = os.path.join(self.cache_dir, 'index.bin')
        self.index_json_file = os.path.join(self.cache_dir, 'index.json')

        # * memory mapped arrays
        self.data = None
        self.indices = None
        self.dataset_info = None

        # * current position trackers
        self.current_train_idx = 0
        self.current_val_idx = 0

        # * section markers (will be initialized)
        self.section_markers = {}

        # * initialize
        self._initialize_markers()
        self._initialize_cache()

    def _initialize_markers(self):
        """initialize and encode section markers"""
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

    def _initialize_cache(self):
        """check if cached files exist, if not create them"""
        cache_exists = (
                os.path.exists(self.data_file) and
                os.path.exists(self.index_file) and
                os.path.exists(self.index_json_file)
        )

        if cache_exists:
            # * check if cache is complete
            with open(self.index_json_file, 'r') as f:
                info = json.load(f)
                if info.get('complete', False):
                    print(f"loading cached data from {self.cache_dir}")
                    self.cache_load()
                else:
                    print(f"incomplete cache found, resuming construction...")
                    self.cache_construct()
                    self.cache_load()
        else:
            print(f"no cached data found, creating cache in {self.cache_dir}")
            self.cache_construct()
            self.cache_load()

    def _save_progress(self, dataset_metadata, total_sequences, total_tokens, max_token_value, complete=False):
        """save progress to index.json"""
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
        """load progress from index.json if exists"""
        if os.path.exists(self.index_json_file):
            with open(self.index_json_file, 'r') as f:
                progress = json.load(f)
                if not progress.get('complete', False):
                    return progress
        return None

    def cache_construct(self):
        """encode datasets and flush to binary files"""
        os.makedirs(self.cache_dir, exist_ok=True)

        # * dataset names to load
        dataset_names = [
            "XenArcAI/CodeX-2M-Thinking",
            "XenArcAI/CodeX-7M-Non-Thinking"
        ]

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

                dataset = load_dataset(dataset_name, split='train', streaming=True)
                dataset_start_seq = total_sequences

                print(f"processing dataset: {dataset_name}")

                processed_sequence = 0

                # * if resuming current dataset, skip already processed sequences
                skip_count = 0
                if dataset_idx == resume_dataset_idx and progress and len(dataset_metadata) > 0:
                    skip_count = dataset_metadata[-1].get('processed_sequence_idx', 0)
                    if skip_count > 0:
                        print(f"skipping first {skip_count} sequences...")

                for row in dataset:
                    # * skip already processed sequences
                    if processed_sequence < skip_count:
                        processed_sequence += 1
                        continue

                    try:
                        tokens = self._process_row(row)

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
                        processed_sequence += 1

                        # * track max token value
                        if len(tokens) > 0:
                            max_token_value = max(max_token_value, int(tokens.max()))

                        if processed_sequence % 1000 == 0:
                            print(f"  processed {processed_sequence:,} sequences, {total_tokens:,} tokens...")

                        # * flush chunk to disk and save progress
                        if len(chunk_tokens) >= chunk_size:
                            chunk_array = np.array(chunk_tokens, dtype=np.uint16)
                            chunk_array.tofile(data_f)
                            chunk_tokens = []

                            # * update or add current dataset metadata
                            current_ds_meta = {
                                'name': dataset_name,
                                'start_sequence_idx': dataset_start_seq,
                                'num_sequences': total_sequences - dataset_start_seq,
                                'processed_sequence_idx': processed_sequence
                            }

                            if dataset_idx == len(dataset_metadata):
                                dataset_metadata.append(current_ds_meta)
                            else:
                                dataset_metadata[dataset_idx] = current_ds_meta

                            # * save progress
                            self._save_progress(dataset_metadata, total_sequences, total_tokens, max_token_value, complete=False)

                    except Exception as e:
                        print(f"error processing row {processed_sequence} from {dataset_name}: {e}")
                        continue

                # * save dataset metadata with final count
                current_ds_meta = {
                    'name': dataset_name,
                    'start_sequence_idx': dataset_start_seq,
                    'num_sequences': total_sequences - dataset_start_seq,
                    'processed_sequence_idx': processed_sequence
                }

                if dataset_idx == len(dataset_metadata):
                    dataset_metadata.append(current_ds_meta)
                else:
                    dataset_metadata[dataset_idx] = current_ds_meta

                print(f"completed {dataset_name}: {processed_sequence:,} sequences")

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

    def cache_load(self):
        """load memory-mapped arrays from cached files"""
        # * load memmap
        self.data = np.memmap(self.data_file, dtype=np.uint16, mode='r')
        self.indices = np.memmap(self.index_file, dtype=np.uint64, mode='r')

        # * load index.json
        with open(self.index_json_file, 'r') as f:
            self.dataset_info = json.load(f)

        print(f"loaded cache: {len(self.data):,} tokens, {len(self.indices):,} sequences")
        print(f"datasets: {len(self.dataset_info['datasets'])}")
        for ds in self.dataset_info['datasets']:
            print(f"  - {ds['name']}: {ds['num_sequences']:,} sequences")

        # * reset position trackers
        self.current_train_idx = 0
        self.current_val_idx = 0

    def _encode_text(self, text: str) -> List[int]:
        """encode text to tokens"""
        if not text:
            return []
        text = text.strip()
        if not text:
            return []
        result = self.nano.encode(text)
        parsed = WordEncodeResult(result)
        return parsed.to_token_list()

    def _extract_language(self, output: str) -> Optional[str]:
        """extract language from code block like ```python"""
        match = re.search(r'```(\w+)', output)
        if match:
            return match.group(1)
        return None

    def _process_thinking(self, output: str) -> tuple[Optional[str], str]:
        """
        extract thinking section if present and return (thinking_content, remaining_output)
        """
        output = output.strip()

        # * check if output starts with <think>
        if output.startswith('<think>'):
            # * find the closing </think>
            end_idx = output.find('</think>')
            if end_idx != -1:
                # * extract thinking content (without <think> tags)
                thinking = output[7:end_idx].strip()  # 7 = len('<think>')
                # * get remaining content after </think>
                remaining = output[end_idx + 8:].strip()  # 8 = len('</think>')
                return thinking, remaining

        return None, output

    def _extract_code(self, text: str) -> str:
        """extract code from markdown code blocks"""
        # * remove code block markers
        text = re.sub(r'```\w+\n?', '', text)
        text = re.sub(r'```', '', text)
        return text.strip()

    def _process_row(self, row: dict) -> Optional[np.ndarray]:
        """process a single row and return tokens"""
        # * get columns
        input_text = row.get('input', '').strip()
        output_text = row.get('output', '').strip()

        if not input_text or not output_text:
            # * skip empty rows
            return None

        # * build token sequence
        tokens = []

        # * 1. instruction section
        tokens.extend(self.section_markers['#sectionInstructionStart#'])
        tokens.extend(self.section_markers['#sectionTemplate1#'])

        # * extract and encode language
        language = self._extract_language(output_text)
        if language:
            tokens.extend(self._encode_text(language))

        tokens.extend(self.section_markers['#sectionInstructionEnd#'])

        # * 2. input section
        tokens.extend(self.section_markers['#sectionInputStart#'])
        tokens.extend(self._encode_text(input_text))
        tokens.extend(self.section_markers['#sectionInputEnd#'])

        # * 3. process thinking and output
        thinking_content, output_text = self._process_thinking(output_text)
        if thinking_content:
            tokens.extend(self.section_markers['#sectionThinkingStart#'])
            tokens.extend(self._encode_text(thinking_content))
            tokens.extend(self.section_markers['#sectionThinkingEnd#'])

        # * 4. output section
        tokens.extend(self.section_markers['#sectionOutputStart#'])

        # * extract and encode code content
        code_content = self._extract_code(output_text)
        tokens.extend(self._encode_text(code_content))

        tokens.extend(self.section_markers['#sectionOutputEnd#'])

        # * convert to numpy array
        return np.array(tokens, dtype=np.uint16)

    def _get_dataset_split_info(self, dataset_idx: int) -> tuple[int, int]:
        """
        get train and val sequence counts for a specific dataset

        Returns:
            (train_seq_count, val_seq_count)
        """
        ds = self.dataset_info['datasets'][dataset_idx]
        total_seqs = ds['num_sequences']
        train_seqs = int(total_seqs * self.train_split)
        val_seqs = total_seqs - train_seqs
        return train_seqs, val_seqs

    def _find_sequence_index(self, split: str) -> Optional[int]:
        """
        find the absolute sequence index for the current train or val position

        Returns:
            absolute sequence index or None if exhausted
        """
        if split == 'train':
            current_pos = self.current_train_idx
        else:
            current_pos = self.current_val_idx

        # * iterate through datasets to find which one contains this position
        cumulative_count = 0
        for dataset_idx, ds in enumerate(self.dataset_info['datasets']):
            train_seqs, val_seqs = self._get_dataset_split_info(dataset_idx)

            if split == 'train':
                target_count = train_seqs
            else:
                target_count = val_seqs

            if current_pos < cumulative_count + target_count:
                # * this dataset contains our position
                local_pos = current_pos - cumulative_count
                dataset_start_seq = ds['start_sequence_idx']

                if split == 'train':
                    absolute_seq_idx = dataset_start_seq + local_pos
                else:
                    # * for val, skip the train portion
                    absolute_seq_idx = dataset_start_seq + train_seqs + local_pos

                return absolute_seq_idx

            cumulative_count += target_count

        # * exhausted all datasets
        return None

    def get(self, split: str = 'train') -> Optional[np.ndarray]:
        """
        get next complete sequence from specified split

        Args:
            split: 'train' or 'val'

        Returns:
            array of tokens for one complete example
        """
        if self.data is None or self.indices is None:
            return None

        # * find absolute sequence index
        seq_idx = self._find_sequence_index(split)

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
        seek to a specific position in the split

        Args:
            position: the position index to seek to (0-based, relative to split)
            split: 'train' or 'val'
        """
        if split == 'train':
            self.current_train_idx = position
        else:
            self.current_val_idx = position

        print(f"seeked to position {position} in {split} split")

    def num_sequences(self, split: str = 'train') -> int:
        """return total number of sequences in specified split"""
        if self.dataset_info is None:
            return 0

        total = 0
        for dataset_idx in range(len(self.dataset_info['datasets'])):
            train_seqs, val_seqs = self._get_dataset_split_info(dataset_idx)
            if split == 'train':
                total += train_seqs
            else:
                total += val_seqs

        return total

    def __len__(self):
        """return total number of tokens"""
        if self.data is not None:
            return len(self.data)
        return 0


def create_loader(cache_dir='.local/cache'):
    """create and return a loader instance"""
    loader = Loader()
    print(f"loader ready: {len(loader):,} total tokens")
    print(f"  train sequences: {loader.num_sequences('train'):,}")
    print(f"  val sequences: {loader.num_sequences('val'):,}")
    return loader


if __name__ == "__main__":
    # * test the loader
    print("creating loader...")
    loader = create_loader()

    print("\ntesting first 3 sequences each split:")
    for i in range(3):
        t = loader.get(split='train')
        if t is not None:
            print(f"\nsequence {i + 1}:")
            print(f"  token count: {len(t):,}")
            print(f"  first 50 tokens: {t[:50]}")
            print(f"  token dtype: {t.dtype}")
        else:
            print(f"\nsequence {i + 1}: no more data")
            break

    for i in range(3):
        t = loader.get(split='val')
        if t is not None:
            print(f"\nsequence {i + 1}:")
            print(f"  token count: {len(t):,}")
            print(f"  first 50 tokens: {t[:50]}")
            print(f"  token dtype: {t.dtype}")
        else:
            print(f"\nsequence {i + 1}: no more data")
            break