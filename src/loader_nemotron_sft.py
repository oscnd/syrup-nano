"""
loader class for leonli66/nemotron-sft-* datasets
"""

import re
import json
import numpy as np
from datasets import load_dataset
from typing import Optional, List
from nano import Nano
from word import WordEncodeResult


class NemotronLoader:
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

    def add_dataset(self, dataset_name: str, subset: str, split: str = 'train'):
        """add a dataset to the loader"""
        print(f"loading dataset: {dataset_name}, subset: {subset}")
        dataset = load_dataset(dataset_name, subset, split=split)

        # * filter for multiturn=false
        dataset = dataset.filter(lambda x: x.get('is_multiturn', True) == False)
        dataset_size = len(dataset)

        self.datasets.append({
            'name': dataset_name,
            'subset': subset,
            'dataset': dataset,
            'size': dataset_size,
            'current_idx': 0
        })

        self.total_items += dataset_size
        print(f"loaded {dataset_size:,} rows from {dataset_name}/{subset} (multiturn=false)")

    def load_datasets(self):
        """load all nemotron datasets"""
        datasets_config = [
            ("leonli66/nemotron-sft-general", "thinking"),
            ("leonli66/nemotron-sft-general", "nonthinking"),
            ("leonli66/nemotron-sft-math", "thinking"),
            ("leonli66/nemotron-sft-math", "nonthinking"),
            ("leonli66/nemotron-sft-code", "thinking"),
            ("leonli66/nemotron-sft-code", "nonthinking"),
        ]

        for dataset_name, subset in datasets_config:
            self.add_dataset(dataset_name, subset)

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

    def _parse_prompt(self, prompt_str: str) -> Optional[str]:
        """
        parse prompt json and extract content from user message
        example input: [{"content": "...", "role": "user"}]
        """
        try:
            prompt_list = json.loads(prompt_str)
            if isinstance(prompt_list, list) and len(prompt_list) > 0:
                # * get first message with role "user"
                for message in prompt_list:
                    if isinstance(message, dict) and message.get('role') == 'user':
                        return message.get('content', '').strip()
        except (json.JSONDecodeError, ValueError) as e:
            print(f"error parsing prompt json: {e}")
        return None

    def _process_target(self, target: str) -> str:
        """
        process target text - remove 'output:' prefix if present and trim
        """
        target = target.strip()
        if target.lower().startswith('output:'):
            target = target[7:].strip()  # remove 'output:' (7 chars)
        return target

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

    def _extract_language(self, text: str) -> Optional[str]:
        """extract language from code block like ```python"""
        match = re.search(r'```(\w+)', text)
        if match:
            return match.group(1)
        return None

    def _process_row(self, row: dict, subset: str) -> Optional[np.ndarray]:
        """process a single row and return tokens"""
        # * get prompt and target
        prompt_str = row.get('prompt', '')
        target_str = row.get('target', '')

        # * parse prompt to get input text
        input_text = self._parse_prompt(prompt_str)
        if not input_text:
            return None

        # * process target to get output text
        output_text = self._process_target(target_str)
        if not output_text:
            return None

        # * build token sequence
        tokens = []

        # * 1. instruction section
        tokens.extend(self.section_markers['#sectionInstructionStart#'])

        # * check if output contains code blocks
        language = self._extract_language(output_text)
        if language:
            tokens.extend(self.section_markers['#sectionTemplate1#'])
            tokens.extend(self._encode_text(language))

        tokens.extend(self.section_markers['#sectionInstructionEnd#'])

        # * 2. input section
        tokens.extend(self.section_markers['#sectionInputStart#'])
        tokens.extend(self._encode_text(input_text))
        tokens.extend(self.section_markers['#sectionInputEnd#'])

        # * 3. process thinking and output (only for thinking subset)
        thinking_content = None
        remaining_output = output_text

        if subset == "thinking":
            thinking_content, remaining_output = self._process_thinking(output_text)

        # * 4. thinking section (if present)
        if thinking_content:
            tokens.extend(self.section_markers['#sectionThinkingStart#'])
            tokens.extend(self._encode_text(thinking_content))
            tokens.extend(self.section_markers['#sectionThinkingEnd#'])

        # * 5. output section
        tokens.extend(self.section_markers['#sectionOutputStart#'])
        tokens.extend(self._encode_text(remaining_output))
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

                # * process row with subset information
                tokens = self._process_row(row, current_ds['subset'])

                if tokens is not None:
                    return tokens
                else:
                    # * skip empty row and continue
                    continue

            except Exception as e:
                print(
                    f"error processing row {current_ds['current_idx']} from {current_ds['name']}/{current_ds['subset']}: {e}")
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
    loader = NemotronLoader()
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