import argparse
import os
from pathlib import Path

import sentencepiece as spm

from data import iter_split_files


def train_unigram(input_dir, model_prefix, vocab_size, character_coverage=1.0, logstream=None):
    """Train a lossless SentencePiece unigram model."""
    # The usual defaults are not reversible: NFKC rewrites characters, repeated
    # spaces are collapsed, and a character missing from training becomes
    # <unk>. Identity normalization keeps each character, whitespace-only
    # pieces keep repeated spaces, and byte fallback encodes anything unseen
    # (including newlines, which are sentence breaks during training) as bytes.
    # input_sentence_size=0 uses every training line. The default length cap
    # would drop the longest lines. character_coverage stays at 1.0 when the
    # vocabulary can hold every character; a smaller sweep has to lower it,
    # and byte fallback still encodes the characters that were left out.
    train_files = [str(path) for path in iter_split_files(input_dir, "train")]
    if not train_files:
        raise ValueError(f"no train.txt files under {input_dir}")
    Path(model_prefix).parent.mkdir(parents=True, exist_ok=True)
    spm.SentencePieceTrainer.Train(
        input=train_files,
        model_prefix=str(model_prefix),
        vocab_size=vocab_size,
        model_type="unigram",
        character_coverage=character_coverage,
        byte_fallback=True,
        normalization_rule_name="identity",
        remove_extra_whitespaces=False,
        allow_whitespace_only_pieces=True,
        unk_id=0,
        bos_id=-1,
        eos_id=-1,
        pad_id=-1,
        input_sentence_size=0,
        max_sentence_length=100000,
        num_threads=os.cpu_count() or 1,
        logstream=logstream,
    )


def main():
    """Train the unigram model and write unigram.model."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--vocab-size", type=int, required=True)
    args = parser.parse_args()
    if args.vocab_size < 256:
        parser.error("--vocab-size must be at least 256")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_unigram(args.input, output_dir / "unigram", args.vocab_size)


if __name__ == "__main__":
    main()
