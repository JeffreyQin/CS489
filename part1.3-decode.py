import argparse
from pathlib import Path


def load_vocab(bpe_dir):
    """Load each token's bytes from bpe_vocab.txt."""
    vocab_path = Path(bpe_dir) / "bpe_vocab.txt"
    return [
        bytes.fromhex(line)
        for line in vocab_path.read_text(encoding="utf-8").splitlines()
    ]


def main():
    """Decode one token-id file back to bytes."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--bpe", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    vocab = load_vocab(args.bpe)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with Path(args.input).open("r", encoding="utf-8") as source, output.open("wb") as handle:
        for line in source:
            handle.write(vocab[int(line)])


if __name__ == "__main__":
    main()
