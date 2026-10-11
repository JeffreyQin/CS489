import argparse
import importlib.util
import json
import sys
from pathlib import Path

from data import load_split, write_ids


def load_script(filename):
    path = Path(__file__).resolve().parent / filename
    spec = importlib.util.spec_from_file_location(filename.replace(".", "_"), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare_corpus(input_dir, output_dir, vocab_size):
    """Train byte-level BPE at this vocabulary size and write train/dev id files.

    The dev text file is the exact string that was encoded, so the character
    count used for bits-per-character matches the token stream.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_dir = output_dir / "tokenizer"

    bpe = load_script("part1.2-bpe.py")
    raw, ends = bpe.load_corpus(input_dir)
    vocab, merges = bpe.train_bpe(raw, ends, vocab_size)
    if len(vocab) != vocab_size:
        raise RuntimeError(f"BPE produced {len(vocab)} tokens, expected {vocab_size}")
    bpe.write_artifacts(tokenizer_dir, vocab, merges)

    encoder = load_script("part1.3-encode.py")
    merge_of = encoder.load_merges(tokenizer_dir)
    train_text = load_split(input_dir, "train")
    dev_text = load_split(input_dir, "dev")
    dev_path = output_dir / "dev.txt"
    dev_path.write_text(dev_text, encoding="utf-8", newline="\n")

    train_ids = encoder.encode(train_text.encode("utf-8"), merge_of)
    dev_ids = encoder.encode(dev_text.encode("utf-8"), merge_of)
    if train_ids and max(train_ids) >= vocab_size:
        raise RuntimeError("train token id is outside the vocabulary")
    if dev_ids and max(dev_ids) >= vocab_size:
        raise RuntimeError("dev token id is outside the vocabulary")
    write_ids(output_dir / "train.bin", train_ids)
    write_ids(output_dir / "dev.bin", dev_ids)
    print(
        f"encoded train={len(train_ids)} dev={len(dev_ids)} tokens",
        file=sys.stderr,
    )
    return dev_path


def write_config(output_dir, vocab_size):
    output_dir = Path(output_dir)
    run_config = json.loads((output_dir / "run_config.json").read_text(encoding="utf-8"))
    run_config["tokenizer"] = {
        "algorithm": "bpe",
        "settings": {
            "vocab_size": vocab_size,
            "byte_level": True,
            "merges_cross_newlines": False,
        },
    }
    (output_dir / "config.json").write_text(
        json.dumps(run_config, indent=2) + "\n",
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--vocab-size", type=int, required=True)
    args = parser.parse_args()
    if args.vocab_size < 256:
        parser.error("--vocab-size must be at least 256")

    dev_path = prepare_corpus(args.input, args.output, args.vocab_size)

    # Imported here so preparing the corpus does not load the training stack
    # until the token files exist.
    from part2_train import train

    metrics = train(
        Path(args.output) / "train.bin",
        Path(args.output) / "dev.bin",
        args.vocab_size,
        args.output,
        dev_text_path=dev_path,
    )
    metrics["vocab_size"] = args.vocab_size
    metrics["algorithm"] = "bpe"
    (Path(args.output) / "dev_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n",
        encoding="utf-8",
    )
    write_config(args.output, args.vocab_size)
    if "dev_bpc" in metrics:
        print(f"dev bpc: {metrics['dev_bpc']:.4f}")


if __name__ == "__main__":
    main()
