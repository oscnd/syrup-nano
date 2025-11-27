"""
Processor for XenArcAI/CodeX-* datasets
"""

import re
import numpy as np
from typing import Optional
from loader_constructor import LoaderConstructorProcessor


class XenarcaiCodexProcessor(LoaderConstructorProcessor):
    """Processor for XenArcAI/CodeX datasets"""

    def can_process(self, dataset_name: str) -> bool:
        """Check if dataset name starts with XenArcAI/CodeX-"""
        return dataset_name.startswith("XenArcAI/CodeX-")

    def should_filter(self, row: dict) -> bool:
        """No filtering needed for CodeX datasets"""
        return True

    def _extract_language(self, output: str) -> Optional[str]:
        """Extract language from code block like ```python"""
        match = re.search(r'```(\w+)', output)
        if match:
            return match.group(1)
        return None

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

    def _extract_code(self, text: str) -> str:
        """Extract code from markdown code blocks"""
        # * remove code block markers
        text = re.sub(r'```\w+\n?', '', text)
        text = re.sub(r'```', '', text)
        return text.strip()

    def process_row(self, row: dict, dataset_name: str) -> Optional[np.ndarray]:
        """Process a single row from XenArcAI CodeX dataset"""
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