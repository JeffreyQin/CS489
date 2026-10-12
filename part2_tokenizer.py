"""
This file was written with assistance from generative AI. Uses include
    - Starter code for the loading, encoding, and decoding process for each of the three tokenizers
      (BPE, SentencePiece, WordPiece (testing)) used to optimize bits-per-character
    - Assistance on code proofreading and improvements

Load a Part 2 tokenizer from a tokenizer directory.
"""

import importlib.util
from pathlib import Path


class ByteBPETokenizer:
    def __init__(self, path):
        """Load vocabulary and merges from path."""
        path = Path(path)
        module = self._encoder()
        self.vocab = [
            bytes.fromhex(line)
            for line in (path / "bpe_vocab.txt").read_text(encoding="utf-8").splitlines()
        ]
        self.vocab_size = len(self.vocab)
        self._encode_bytes = module.encode
        self.merge_of = module.load_merges(path)

    def _encoder(self):
        """load encoder from part1.3-encode.py"""
        path = Path(__file__).resolve().parent / "part1.3-encode.py"
        spec = importlib.util.spec_from_file_location("part1_3_encode", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def encode(self, text):
        return self._encode_bytes(text.encode("utf-8"), self.merge_of)

    def decode(self, token_ids):
        return _decode_vocab(self.vocab, token_ids)


class UnigramTokenizer:
    def __init__(self, path):
        """Load the SentencePiece unigram model from path."""
        import sentencepiece as spm

        self.model = spm.SentencePieceProcessor()
        self.model.Load(str(Path(path) / "unigram.model"))
        self.vocab_size = self.model.GetPieceSize()

    def encode(self, text):
        return self.model.Encode(text)

    def decode(self, token_ids):
        return self.model.Decode(list(token_ids))


class WordPieceTokenizer:
    def __init__(self, path):
        """Load the WordPiece vocabulary from path."""
        self.vocab = [
            bytes.fromhex(line)
            for line in (Path(path) / "wordpiece_vocab.txt").read_text(encoding="utf-8").splitlines()
        ]
        self.vocab_size = len(self.vocab)
        # trie of vocabulary bytes. -1 marks the end of a token.
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
        # newline is token 10 and does not join the text around it.
        for index, byte in enumerate(data):
            if byte != 10:
                continue
            if index > start:
                out.extend(self._encode_span(data, start, index))
            out.append(10)
            start = index + 1
        # encode a final line that has no newline.
        if start < len(data):
            out.extend(self._encode_span(data, start, len(data)))
        return out

    def _encode_span(self, data, start, end):
        out = []
        index = start
        root = self._root
        while index < end:
            node = root
            best = None
            best_end = index
            cursor = index
            # keep the longest vocabulary token that starts here.
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
        return _decode_vocab(self.vocab, token_ids)


def _decode_vocab(vocab, token_ids):
    """Join vocabulary bytes and decode them as text."""
    return b"".join(vocab[token_id] for token_id in token_ids).decode("utf-8")


def load_tokenizer(path):
    """Reload the tokenizer written under path."""
    path = Path(path)
    if (path / "unigram.model").is_file():
        return UnigramTokenizer(path)
    if (path / "wordpiece_vocab.txt").is_file():
        return WordPieceTokenizer(path)
    return ByteBPETokenizer(path)
