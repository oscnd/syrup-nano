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

        # load shared library
        self.lib = ctypes.PyDLL(library_path)

        # define function signatures
        self.lib.load.argtypes = []
        self.lib.load.restype = None

        self.lib.encode.argtypes = [ctypes.c_char_p]
        self.lib.encode.restype = ctypes.c_char_p

        self.lib.decode.argtypes = [ctypes.c_ulonglong]
        self.lib.decode.restype = ctypes.c_char_p

        # library initialization
        self.lib.load()

    def encode(self, text: str) -> list:
        # convert python string to c string
        c_text = ctypes.create_string_buffer(text.encode('utf-8'))

        # call encode function
        result = self.lib.encode(c_text)

        # convert c string result to python string
        result_str = ctypes.string_at(result).decode('utf-8')

        try:
            # parse json result
            return json.loads(result_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"failed to decode json: {e}")

    def decode(self, token: int) -> str:
        # convert python int to c_ulonglong
        c_token = ctypes.c_ulonglong(token)

        # call decode function
        result = self.lib.decode(c_token)

        # convert c string result to python string
        result_str = ctypes.string_at(result).decode('utf-8')

        return result_str
