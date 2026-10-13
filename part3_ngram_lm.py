"""
This file was written with assistance from generative AI. 
All AI-generated code were read-over and improved by hand for correctness.

Uses of AI include
    - Starter code for argument parsing and file path handling
    - Implementation of critical helper utilities for corpus processing, such as
      tokenize_lines, dev_char_count, iter_raw_lines
    - Starter code and assistance with proofreading for the algorithm, such as 
      train_counts, format_gram, and dev_perplexity (eval util)
"""


import argparse
import importlib.util
import math
import sys
from pathlib import Path

from data import iter_split_files


def load_encoder(bpe_dir):
    """Load byte-level BPE encoder, merges, and vocabulary."""

    path = Path(__file__).resolve().parent / "part1.3-encode.py"
    spec = importlib.util.spec_from_file_location("part1_3_encode", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    bpe_dir = Path(bpe_dir)

    # one hex token per line, in id order.
    vocab = [
        bytes.fromhex(line)
        for line in (bpe_dir / "bpe_vocab.txt").read_text(encoding="utf-8").splitlines()
    ]
    return module.encode_line, module.load_merges(bpe_dir), vocab


def iter_raw_lines(input_dir, split):
    """Yield each line of a split as bytes, without the newline."""
    for path in iter_split_files(input_dir, split):
        data = path.read_bytes()
        if not data:
            continue
        parts = data.split(b"\n")

        # drop empty piece left by a trailing newline.
        if data.endswith(b"\n"):
            parts.pop()
        yield from parts


def tokenize_line(line, encode_line, merge_of, eos_id):
    """Encode one line and append the end-of-line token."""
    ids = encode_line(line, merge_of)
    ids.append(eos_id)
    return ids


def pack(token_ids):
    """Pack token ids into one integer key."""
    key = 0
    for token_id in token_ids:
        key = (key << 16) | token_id
    return key


def train_counts(lines, order):
    """Count n-grams up to the given order."""

    # counts[k] stores k-grams. index 0 is unused.
    counts = [None] + [{} for _ in range(order)]
    total = 0
    for tokens in lines:
        total += len(tokens)
        for index in range(len(tokens)):
            # longest gram ending here, then each shorter one down to one token.
            longest = min(order, index + 1)
            for length in range(longest, 0, -1):
                key = pack(tokens[index + 1 - length : index + 1])
                counts[length][key] = counts[length].get(key, 0) + 1
    return counts, total


def dev_perplexity(lines, order, counts, total, alpha=0.0, vocab_size=None):
    """Score dev perplexity, with optional add-alpha smoothing."""
    if alpha < 0:
        raise ValueError("alpha must be >= 0")
    if alpha > 0 and vocab_size is None:
        raise ValueError("add-alpha scoring needs vocab_size")
    log_sum = 0.0
    scored = 0
    for tokens in lines:
        for index, token_id in enumerate(tokens):
            context_len = min(order - 1, index)
            context = tokens[index - context_len : index]
            gram = context + [token_id]
            numerator = counts[context_len + 1].get(pack(gram), 0)
            # an empty context is counted as the number of training tokens.
            if context_len == 0:
                denominator = total
            else:
                denominator = counts[context_len].get(pack(context), 0)
            scored += 1
            # one unseen gram makes the perplexity infinite.
            if alpha == 0 and (numerator == 0 or denominator == 0):
                return math.inf, gram, scored
            if alpha == 0:
                log_sum += math.log2(numerator / denominator)
            else:
                log_sum += math.log2(numerator + alpha) - math.log2(
                    denominator + alpha * vocab_size
                )
    if scored == 0:
        raise ValueError("dev set has no tokens")
    return 2 ** (-log_sum / scored), None, scored


def format_gram(token_ids, vocab, eos_id):
    """Format one n-gram for an error message."""
    pieces = []
    for token_id in token_ids:
        if token_id == eos_id:
            pieces.append("<EOS>")
        elif token_id < len(vocab):
            pieces.append(repr(vocab[token_id].decode("utf-8", "replace")))
        else:
            pieces.append(str(token_id))
    ids = " ".join(str(token_id) for token_id in token_ids)
    return f"{' '.join(pieces)} (ids {ids})"


def dev_char_count(input_dir):
    """Count Unicode characters in the cleaned dev text."""
    return sum(
        len(path.read_text(encoding="utf-8"))
        for path in iter_split_files(input_dir, "dev")
    )


def main():
    """
    Train n-gram model at a given order
    print dev perplexity and bits per character.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--bpe", type=str, required=True)
    parser.add_argument("--order", type=int, required=True)
    parser.add_argument("--smoothing", type=str, required=True)
    parser.add_argument("--alpha", type=float)
    args = parser.parse_args()
   
    if args.smoothing == "none":
        assert(args.alpha is None)
        alpha = 0.0
    elif args.smoothing == "add-alpha":
        assert(args.alpha is not None)
        alpha = args.alpha

    # load vocabulary from Part 1.3 encoder
    encode_line, merge_of, vocab = load_encoder(args.bpe)

    # end-of-line id is one past the last bpe id.
    eos_id = len(vocab)
    vocab_size = eos_id + 1

    def token_lines(split):
        """Tokenize each line of a split and append the end-of-line token."""
        for line in iter_raw_lines(args.input, split):
            yield tokenize_line(line, encode_line, merge_of, eos_id)

    # train n-gram model
    print(f"training order-{args.order} counts", file=sys.stderr)
    counts, total = train_counts(token_lines("train"), args.order)
    print(f"train tokens={total}", file=sys.stderr)

    # compute perplexity on dev set
    perplexity, missing, scored = dev_perplexity(
        token_lines("dev"),
        args.order,
        counts,
        total,
        alpha=alpha,
        vocab_size=vocab_size,
    )
    n_chars = dev_char_count(args.input)

    # if no missing grams, compute bits per character
    # otherwise perplexity is infinite
    if missing is None:
        bpc = scored / n_chars * math.log2(perplexity)
        print(f"dev perplexity={perplexity:.6f}")
        print(
            f"dev tokens={scored} dev chars={n_chars} dev bpc={bpc:.4f}",
            file=sys.stderr,
        )
    else:
        print("dev perplexity=inf")
        print(
            f"dev perplexity is infinite; unseen {len(missing)}-gram "
            f"{format_gram(missing, vocab, eos_id)}",
            file=sys.stderr,
        )
        print(f"dev bpc=inf ({n_chars} chars)", file=sys.stderr)


if __name__ == "__main__":
    main()
