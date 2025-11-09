#!/usr/bin/env python3
import ast

class WordPair:
    def __init__(self, word, token):
        self.word = word
        self.token = token

    def __repr__(self):
        return f"WordPair(word={self.word!r}, token={self.token})"

    def to_dict(self):
        return {'word': self.word, 'token': self.token}


class WordEncodeResult:
    def __init__(self, result):
        self.word_pairs = [WordPair(item['word'], item['token'])
                           for item in result]

    def __repr__(self):
        return f"WordEncodeResult({len(self.word_pairs)} tokens)"

    def __iter__(self):
        return iter(self.word_pairs)

    def __len__(self):
        return len(self.word_pairs)

    def __getitem__(self, idx):
        return self.word_pairs[idx]

    def to_list(self):
        return self.word_pairs
