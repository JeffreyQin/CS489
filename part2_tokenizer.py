"""Part-2 tokenizer: byte-level BPE, same rules as part1.3-encode.py.

load_tokenizer(path) reads bpe_vocab.txt and bpe_merges.txt from path.
encode(text) returns token ids in {0, ..., V - 1}. Newlines stay token 10
and are never inside a longer token, matching how the vocabulary was trained.
Decoding the ids reproduces the input text, including line breaks.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _encoder():
    path = Path(__file__).resolve().parent / "part1.3-encode.py"
    spec = importlib.util.spec_from_file_location("part1_3_encode", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ByteBPETokenizer:
    def __init__(self, path):
        path = Path(path)
        module = _encoder()
        self.vocab = [
            bytes.fromhex(line)
            for line in (path / "bpe_vocab.txt").read_text(encoding="utf-8").splitlines()
        ]
        self._encode_bytes = module.encode
        self.merge_of = module.load_merges(path)

    def encode(self, text):
        return self._encode_bytes(text.encode("utf-8"), self.merge_of)

    def decode(self, token_ids):
        return b"".join(self.vocab[token_id] for token_id in token_ids).decode("utf-8")


def load_tokenizer(path: str):
    """Reload the byte-level BPE tokenizer written under path."""
    return ByteBPETokenizer(path)
