"""
This file was written with assistance from generative AI. 
All AI-generated code were read-over and improved by hand for correctness.

Uses of AI include
    - primary implementation of all functions used for loading, copying and saving training artifacts,
      including copy_tokenizer, saved_tokenizer_dir, write_config, and prepare_corpus
    - Code for argument parsing and file path handling
    - Code proofreading and improvements
"""


import argparse
import json
import shutil
import sys
from pathlib import Path

from data import load_split, write_ids
from part2_tokenizer import load_tokenizer


ROOT = Path(__file__).resolve().parent


def saved_tokenizer_dir(algorithm, vocab_size):
    """Return the Part-1 directory that already holds this vocabulary."""
    if algorithm == "bpe" and vocab_size == 50000:
        submitted = ROOT / "artifacts" / "part1" / "bpe"
        if submitted.is_dir():
            return submitted
    source = ROOT / "artifacts" / "part1" / "analysis" / algorithm / f"v{vocab_size}"
    if source.is_dir():
        return source
    if algorithm == "unigram" and vocab_size == 50000:
        submitted = ROOT / "artifacts" / "part1" / "unigram"
        if submitted.is_dir():
            return submitted
    raise FileNotFoundError(
        f"no saved {algorithm} vocabulary of size {vocab_size} under artifacts/part1"
    )


def tokenizer_algorithm(source):
    """Return bpe, unigram, or wordpiece based on the files in source."""
    source = Path(source)
    if (source / "unigram.model").is_file():
        return "unigram"
    if (source / "wordpiece_vocab.txt").is_file():
        return "wordpiece"
    if (source / "bpe_vocab.txt").is_file() and (source / "bpe_merges.txt").is_file():
        return "bpe"
    raise FileNotFoundError(f"no BPE, unigram, or WordPiece tokenizer in {source}")


def copy_tokenizer(source, dest, algorithm):
    """Copy the saved tokenizer files into dest."""
    source = Path(source)
    dest = Path(dest)
    if dest.exists():
        for child in dest.iterdir():
            if child.is_file():
                child.unlink()
    dest.mkdir(parents=True, exist_ok=True)
    names = {
        "bpe": ("bpe_vocab.txt", "bpe_merges.txt"),
        "wordpiece": ("wordpiece_vocab.txt",),
        "unigram": ("unigram.model",),
    }[algorithm]
    for name in names:
        src = source / name
        if not src.is_file():
            raise FileNotFoundError(f"missing {src}")
        shutil.copy2(src, dest / name)
    extra_name = {"unigram": "unigram.vocab", "wordpiece": "wordpiece_merges.txt"}.get(algorithm)
    extra = source / extra_name if extra_name else None
    if extra is not None and extra.is_file():
        shutil.copy2(extra, dest / extra_name)


def prepare_corpus(input_dir, output_dir, vocab_size, algorithm, tokenizer_path):
    """Copy a saved tokenizer, encode train and dev, and write the id files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source = Path(tokenizer_path) if tokenizer_path else saved_tokenizer_dir(algorithm, vocab_size)
    if not source.is_dir():
        raise FileNotFoundError(f"tokenizer directory does not exist: {source}")
    tokenizer_dir = output_dir / "tokenizer"
    copy_tokenizer(source, tokenizer_dir, algorithm)

    tokenizer = load_tokenizer(str(tokenizer_dir))
    if tokenizer.vocab_size != vocab_size:
        raise RuntimeError(
            f"{algorithm} vocabulary has {tokenizer.vocab_size} pieces, expected {vocab_size}"
        )

    train_text = load_split(input_dir, "train")
    dev_text = load_split(input_dir, "dev")
    dev_path = output_dir / "dev.txt"
    dev_path.write_text(dev_text, encoding="utf-8", newline="\n")
    train_ids = tokenizer.encode(train_text)
    dev_ids = tokenizer.encode(dev_text)
    if tokenizer.decode(dev_ids) != dev_text:
        raise RuntimeError("dev encoding is not lossless")
    write_ids(output_dir / "train.bin", train_ids)
    write_ids(output_dir / "dev.bin", dev_ids)
    print(f"loaded {algorithm} V={vocab_size} from {source}", file=sys.stderr)
    print(f"encoded train={len(train_ids)} dev={len(dev_ids)} tokens", file=sys.stderr)
    return dev_path, source


def write_config(output_dir, vocab_size, algorithm, source):
    """Add the tokenizer section to config.json."""
    output_dir = Path(output_dir)
    run_config = json.loads((output_dir / "run_config.json").read_text(encoding="utf-8"))
    if algorithm == "unigram":
        settings = {
            "vocab_size": vocab_size,
            "model_type": "unigram",
            "byte_fallback": True,
            "normalization": "identity",
            "source": str(source),
        }
    else:
        settings = {
            "vocab_size": vocab_size,
            "byte_level": True,
            "merges_cross_newlines": False,
        }
        if algorithm == "wordpiece":
            settings.update(
                {
                    "score": "count(pair) / (count(left) * count(right))",
                    "min_pair_count": 50,
                    "character_min_freq": 50,
                    "symbols": "bytes plus multibyte characters that occur at least 50 times",
                    "word_split": "space and tab",
                    "encode": "longest_match",
                }
            )
        settings["source"] = str(source)
    run_config["tokenizer"] = {"algorithm": algorithm, "settings": settings}
    (output_dir / "config.json").write_text(
        json.dumps(run_config, indent=2) + "\n",
        encoding="utf-8",
    )


def main():
    """Load a saved tokenizer, train the fixed Transformer, and write the artifacts."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--algorithm", choices=("bpe", "unigram", "wordpiece"), default="bpe")
    parser.add_argument("--tokenizer", type=str, default=None)
    args = parser.parse_args()
    if args.vocab_size < 256:
        parser.error("--vocab-size must be at least 256")

    algorithm = args.algorithm
    if args.tokenizer:
        found = tokenizer_algorithm(args.tokenizer)
        if found != algorithm:
            parser.error(f"{args.tokenizer} is a {found} tokenizer, not {algorithm}")

    dev_path, source = prepare_corpus(
        args.input, args.output, args.vocab_size, algorithm, args.tokenizer
    )

    # import after the corpus exists so training is not loaded earlier.
    from part2_train import train

    metrics = train(
        Path(args.output) / "train.bin",
        Path(args.output) / "dev.bin",
        args.vocab_size,
        args.output,
        dev_text_path=dev_path,
    )
    metrics["vocab_size"] = args.vocab_size
    metrics["algorithm"] = algorithm
    (Path(args.output) / "dev_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n",
        encoding="utf-8",
    )
    write_config(args.output, args.vocab_size, algorithm, source)
    if "dev_bpc" in metrics:
        print(f"dev bpc: {metrics['dev_bpc']:.4f}")


if __name__ == "__main__":
    main()
