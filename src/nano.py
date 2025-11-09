#!/usr/bin/env python3
import ctypes
import json
import os
from pathlib import Path

project_root = Path(__file__).parent.parent
nano_so_path = project_root / "nano.so"

class Nano:
    def __init__(self, library_path: str = None):
        if library_path is None:
            library_path = str(nano_so_path)

        if not os.path.exists(library_path):
            raise FileNotFoundError(f"nano.so not found at {library_path}")

        # Load the shared library
        self.lib = ctypes.PyDLL(library_path)

        # Define function signatures
        self.lib.load.argtypes = []
        self.lib.load.restype = None

        self.lib.encode.argtypes = [ctypes.c_char_p]
        self.lib.encode.restype = ctypes.c_char_p

        # Load the library
        self.lib.load()

    def encode(self, text: str) -> list:
        """
        Encode text using the nano.so encode function

        Args:
            text: Input text to encode

        Returns:
            List of word pairs/tokens from the tokenizer
        """
        # Convert Python string to C string
        c_text = ctypes.create_string_buffer(text.encode('utf-8'))

        # Call the encode function
        result = self.lib.encode(c_text)

        # Convert C string back to Python string
        result_str = ctypes.string_at(result).decode('utf-8')

        try:
            # Parse JSON result into Python list
            return json.loads(result_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON result: {e}\nRaw result: {result_str}")
