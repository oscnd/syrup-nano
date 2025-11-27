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

        self.lib.free_cstring.argtypes = [ctypes.c_void_p]
        self.lib.free_cstring.restype = None

        self.lib.encode.argtypes = [ctypes.c_char_p]
        self.lib.encode.restype = ctypes.c_void_p

        self.lib.decode.argtypes = [ctypes.c_ulonglong]
        self.lib.decode.restype = ctypes.c_void_p

        self.lib.get_num.argtypes = []
        self.lib.get_num.restype = ctypes.c_ulonglong

        self.lib.construct_word_special.argtypes = [ctypes.c_char_p]
        self.lib.construct_word_special.restype = None

        self.lib.construct_from_glob.argtypes = [ctypes.c_char_p]
        self.lib.construct_from_glob.restype = None

        self.lib.construct_content.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
        self.lib.construct_content.restype = None

        self.lib.construct_from_file.argtypes = [ctypes.c_char_p]
        self.lib.construct_from_file.restype = None

        self.lib.shutdown.argtypes = []
        self.lib.shutdown.restype = None

        # library initialization
        self.lib.load()

    def encode(self, text: str) -> list:
        # convert python string to c string
        c_text = ctypes.create_string_buffer(text.encode('utf-8'))

        # call encode function
        result_ptr = self.lib.encode(c_text)

        # convert c string result to python string
        result_str = ctypes.cast(result_ptr, ctypes.c_char_p).value.decode('utf-8')

        # free c string
        self.lib.free_cstring(result_ptr)

        try:
            # parse json result
            return json.loads(result_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"failed to decode json: {e}")

    def decode(self, token: int) -> str:
        # convert python int to c_ulonglong
        c_token = ctypes.c_ulonglong(token)

        # call decode function
        result_ptr = self.lib.decode(c_token)

        # convert c string result to python string
        result_str = ctypes.cast(result_ptr, ctypes.c_char_p).value.decode('utf-8')

        # free c string
        self.lib.free_cstring(result_ptr)

        return result_str

    def get_num(self) -> int:
        """Get current token counter value"""
        result = self.lib.get_num()
        return result

    def construct_word_special(self, pattern: str):
        """Construct word special tokens from files matching the glob pattern"""
        c_pattern = ctypes.create_string_buffer(pattern.encode('utf-8'))
        self.lib.construct_word_special(c_pattern)

    def construct_from_glob(self, pattern: str):
        """Construct tokens from JSONL files matching the glob pattern"""
        c_pattern = ctypes.create_string_buffer(pattern.encode('utf-8'))
        self.lib.construct_from_glob(c_pattern)

    def construct_from_file(self, filename: str):
        """Process a file as content"""
        c_filename = ctypes.create_string_buffer(filename.encode('utf-8'))
        self.lib.construct_from_file(c_filename)

    def construct_content(self, filename: str, content: str):
        """Construct tokens from raw content string"""
        c_filename = ctypes.create_string_buffer(filename.encode('utf-8'))
        c_content = ctypes.create_string_buffer(content.encode('utf-8'))
        self.lib.construct_content(c_filename, c_content)

    def shutdown(self):
        """Shutdown the nano application"""
        self.lib.shutdown()
