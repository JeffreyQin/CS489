"""
This file was written with assistance from generative AI. 
All AI-generated code were read-over and improved by hand for correctness.

Uses of AI include
    - Starter code for argument parsing and file path handling
    - Autocompletion assistance and proofreading
"""


import argparse
import sys
from pathlib import Path

from part3_artifacts import ngram_key, save_ngram_model
from part3_ngram_lm import iter_raw_lines, load_encoder, tokenize_line, train_counts

# Lowest dev perplexity on the order/alpha sweep: bigrams, add-alpha 0.004.
ORDER = 2
ALPHA = 0.01


def unpack(key, length):
    """Turn a packed n-gram key back into token ids."""
    token_ids = [0] * length
    for index in range(length - 1, -1, -1):
        token_ids[index] = key & 0xFFFF
        key >>= 16
    return token_ids


def main():
    """Train the selected n-gram model and write model.json.gz."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--bpe", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    # load vocabulary from part 1.3 encoder
    encode_line, merge_of, vocab = load_encoder(args.bpe)

    # add end-of-line id
    eos_id = len(vocab)
    vocab_size = eos_id + 1

    def token_lines():
        """Tokenize each training line and append the end-of-line token."""
        for line in iter_raw_lines(args.input, "train"):
            yield tokenize_line(line, encode_line, merge_of, eos_id)

    # train n-gram model at the selected order
    print(f"training order-{ORDER} add-alpha {ALPHA}", file=sys.stderr)
    counts, total = train_counts(token_lines(), ORDER)
    print(f"train tokens={total}", file=sys.stderr)

    tables = {}
    for length in range(1, ORDER + 1):
        table = {}
        # turn each packed key back into space-joined token ids.
        for key, count in counts[length].items():
            table[ngram_key(unpack(key, length))] = count
        tables[str(length)] = table
        print(f"order-{length} types={len(table)}", file=sys.stderr)

    # save model to model.json.gz inside output directory
    output = Path(args.output) / "model.json.gz"
    save_ngram_model(
        {
            "n": ORDER,
            "alpha": ALPHA,
            "vocab_size": vocab_size,
            "counts": tables,
        },
        output,
    )
    print(f"wrote {output}", file=sys.stderr)


if __name__ == "__main__":
    main()
