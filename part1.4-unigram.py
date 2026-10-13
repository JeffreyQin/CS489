"""
This file was written with assistance from generative AI. 
All AI-generated code were read-over and improved by hand for correctness.

Uses of AI include
    - Code for argument parsing and file path handling
    - Primary implementation of train_unigram, including usage of the sentencepiece library
"""

import argparse
import os
from pathlib import Path

import sentencepiece as spm

from data import iter_split_files


def train_unigram(input_dir, model_prefix, vocab_size, character_coverage=1.0, logstream=None):
    """Train lossless SentencePiece unigram model."""

    train_files = [str(path) for path in iter_split_files(input_dir, "train")]

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
