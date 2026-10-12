"""Part-2 tokenizer loader.

load_tokenizer(path) reads tokenizer/. unigram.model loads SentencePiece.
wordpiece_vocab.txt loads byte-level WordPiece. Otherwise bpe_vocab.txt and
bpe_merges.txt load byte-level BPE. encode(text) returns token ids in
{0, ..., V - 1}. decode(encode(text)) reproduces the text, including line breaks.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _encoder():
    """Load the Part 1.3 encoder."""
    path = Path(__file__).resolve().parent / "part1.3-encode.py"
    spec = importlib.util.spec_from_file_location("part1_3_encode", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ByteBPETokenizer:
    def __init__(self, path):
        """Load the vocabulary and merges from path."""
        path = Path(path)
        module = _encoder()
        self.vocab = [
            bytes.fromhex(line)
            for line in (path / "bpe_vocab.txt").read_text(encoding="utf-8").splitlines()
        ]
        self.vocab_size = len(self.vocab)
        self._encode_bytes = module.encode
        self.merge_of = module.load_merges(path)

    def encode(self, text):
        """Encode text to token ids."""
        return self._encode_bytes(text.encode("utf-8"), self.merge_of)

    def decode(self, token_ids):
        """Decode token ids back to text."""
        return b"".join(self.vocab[token_id] for token_id in token_ids).decode("utf-8")


class UnigramTokenizer:
    def __init__(self, path):
        """Load the SentencePiece unigram model from path."""
        import sentencepiece as spm

        self.model = spm.SentencePieceProcessor()
        self.model.Load(str(Path(path) / "unigram.model"))
        self.vocab_size = self.model.GetPieceSize()

    def encode(self, text):
        """Encode text to token ids."""
        return self.model.Encode(text)

    def decode(self, token_ids):
        """Decode token ids back to text."""
        return self.model.Decode(list(token_ids))


class WordPieceTokenizer:
    def __init__(self, path):
        """Load the WordPiece vocabulary from path."""
        self.vocab = [
            bytes.fromhex(line)
            for line in (Path(path) / "wordpiece_vocab.txt").read_text(encoding="utf-8").splitlines()
        ]
        self.vocab_size = len(self.vocab)
        root = {}
        for token_id, token in enumerate(self.vocab):
            node = root
            for byte in token:
                child = node.get(byte)
                if child is None:
                    child = {}
                    node[byte] = child
                node = child
            node[-1] = token_id
        self._root = root

    def encode(self, text):
        """Encode text to token ids with longest match."""
        data = text.encode("utf-8")
        out = []
        start = 0
        for index, byte in enumerate(data):
            if byte != 10:
                continue
            if index > start:
                out.extend(self._encode_span(data, start, index))
            out.append(10)
            start = index + 1
        if start < len(data):
            out.extend(self._encode_span(data, start, len(data)))
        return out

    def _encode_span(self, data, start, end):
        """Encode one line of bytes."""
        out = []
        index = start
        root = self._root
        while index < end:
            node = root
            best = None
            best_end = index
            cursor = index
            while cursor < end:
                node = node.get(data[cursor])
                if node is None:
                    break
                cursor += 1
                token_id = node.get(-1)
                if token_id is not None:
                    best = token_id
                    best_end = cursor
            if best is None:
                raise RuntimeError("byte missing from the WordPiece vocabulary")
            out.append(best)
            index = best_end
        return out

    def decode(self, token_ids):
        """Decode token ids back to text."""
        return b"".join(self.vocab[token_id] for token_id in token_ids).decode("utf-8")


def load_tokenizer(path: str):
    """Reload the tokenizer written under path."""
    path = Path(path)
    if (path / "unigram.model").is_file():
        return UnigramTokenizer(path)
    if (path / "wordpiece_vocab.txt").is_file():
        return WordPieceTokenizer(path)
    return ByteBPETokenizer(path)
