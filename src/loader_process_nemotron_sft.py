"""
Processor for leonli66/nemotron-sft-* datasets
"""

import re
import json
import numpy as np
from typing import Optional
from loader_constructor import LoaderConstructorProcessor


class NemotronSftProcessor(LoaderConstructorProcessor):
    """Processor for leonli66/nemotron-sft datasets"""

    def can_process(self, dataset_name: str) -> bool:
        """Check if dataset name starts with leonli66/nemotron-sft-"""
        return dataset_name.startswith("leonli66/nemotron-sft-")

    def should_filter(self, row: dict) -> bool:
        """Filter for multiturn=false only"""
        return row.get('is_multiturn', True) == False

    def _parse_prompt(self, prompt_str: str) -> Optional[str]:
        """
        Parse prompt json and extract content from user message
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
        Process target text - remove 'output:' prefix if present and trim
        """
        target = target.strip()
        if target.lower().startswith('output:'):
            target = target[7:].strip()  # remove 'output:' (7 chars)
        return target

    def _process_thinking(self, output: str) -> tuple[Optional[str], str]:
        """
        Extract thinking section if present and return (thinking_content, remaining_output)
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
        """Extract language from code block like ```python"""
        match = re.search(r'```(\w+)', text)
        if match:
            return match.group(1)
        return None

    def _is_thinking_subset(self, dataset_name: str) -> bool:
        """Check if the dataset is a 'thinking' subset"""
        # Extract subset from dataset name like "leonli66/nemotron-sft-general/thinking"
        # For now, we'll check if 'thinking' appears in the name
        return 'thinking' in dataset_name.lower()

    def process_row(self, row: dict, dataset_name: str) -> Optional[np.ndarray]:
        """Process a single row from Nemotron SFT dataset"""
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

        if self._is_thinking_subset(dataset_name):
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