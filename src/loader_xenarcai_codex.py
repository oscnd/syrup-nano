"""
loader class for dataset XenArcAI/CodeX-*
"""

import re
import numpy as np
from datasets import load_dataset
from typing import Optional, List
from nano import Nano
from word import WordEncodeResult


class Loader:
    def __init__(self):
        self.datasets = []
        self.current_dataset_idx = 0
        self.current_idx = 0
        self.total_items = 0
        self.nano = Nano()

        # * define section markers as tokens
        self.section_markers = {
            '#sectionInstructionStart#': [0],
            '#sectionTemplate1#': [0],
            '#sectionInstructionEnd#': [0],
            '#sectionInputStart#': [0],
            '#sectionInputEnd#': [0],
            '#sectionThinkingStart#': [0],
            '#sectionThinkingEnd#': [0],
            '#sectionOutputStart#': [0],
            '#sectionOutputEnd#': [0],
        }

        # * encode section markers
        for marker in self.section_markers.keys():
            result = self.nano.encode(marker)
            parsed = WordEncodeResult(result)
            self.section_markers[marker] = parsed.to_token_list()

    def add_dataset(self, dataset_name: str):
        """add a dataset to the loader"""
        print(f"loading dataset: {dataset_name}")
        dataset = load_dataset(dataset_name)
        dataset_size = len(dataset)

        self.datasets.append({
            'name': dataset_name,
            'dataset': dataset,
            'size': dataset_size,
            'current_idx': 0
        })

        self.total_items += dataset_size
        print(f"loaded {dataset_size:,} rows from {dataset_name}")

    def load_datasets(self):
        """load all codex datasets"""
        self.add_dataset("XenArcAI/CodeX-2M-Thinking")
        self.add_dataset("XenArcAI/CodeX-7M-Non-Thinking")
        print(f"\ntotal entries across all datasets: {self.total_items:,}")

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
        thinking_content, remaining_output = self._process_thinking(output_text)

        # * 4. thinking section (if present)
        if thinking_content:
            tokens.extend(self.section_markers['#sectionThinkingStart#'])
            tokens.extend(self._encode_text(thinking_content))
            tokens.extend(self.section_markers['#sectionThinkingEnd#'])

        # * 5. output section
        tokens.extend(self.section_markers['#sectionOutputStart#'])

        # * extract and encode code content
        code_content = self._extract_code(remaining_output)
        tokens.extend(self._encode_text(code_content))

        tokens.extend(self.section_markers['#sectionOutputEnd#'])

        # * convert to numpy array
        return np.array(tokens, dtype=np.uint16)

    def get(self) -> Optional[np.ndarray]:
        """
        get next row as numpy array of uint16 tokens
        returns none when all datasets are exhausted
        """
        if not self.datasets:
            self.load_datasets()

        # * iterate through datasets
        while self.current_dataset_idx < len(self.datasets):
            current_ds = self.datasets[self.current_dataset_idx]

            # * check if current dataset is exhausted
            if current_ds['current_idx'] >= current_ds['size']:
                # * move to next dataset
                self.current_dataset_idx += 1
                continue

            try:
                # * get row from current dataset
                row = current_ds['dataset'][current_ds['current_idx']]
                current_ds['current_idx'] += 1
                self.current_idx += 1

                # * process row
                tokens = self._process_row(row)

                if tokens is not None:
                    return tokens
                else:
                    # * skip empty row and continue
                    continue

            except Exception as e:
                print(f"error processing row {current_ds['current_idx']} from {current_ds['name']}: {e}")
                current_ds['current_idx'] += 1
                self.current_idx += 1
                continue

        # * all datasets exhausted
        return None

    def reset(self):
        """reset iterator to beginning"""
        self.current_dataset_idx = 0
        self.current_idx = 0
        for ds in self.datasets:
            ds['current_idx'] = 0

    def __len__(self):
        if not self.datasets:
            self.load_datasets()
        return self.total_items


def create_loader():
    """create and return a loader instance"""
    loader = Loader()
    print(f"total entries in loader: {len(loader):,}")
    return loader


if __name__ == "__main__":
    # * test the loader
    loader = create_loader()

    print("\ntesting first 3 rows:")
    for i in range(3):
        tokens = loader.get()
        if tokens is not None:
            print(f"\nrow {i + 1}:")
            print(f"  token count: {len(tokens):,}")
            print(f"  first 50 tokens: {tokens[:50]}")
            print(f"  token dtype: {tokens.dtype}")
        else:
            print(f"\nrow {i + 1}: no more data")
            break