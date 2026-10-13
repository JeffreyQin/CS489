"""
This file was written with assistance from generative AI. 
All AI-generated code were read-over and improved by hand for correctness.

Uses of AI include
    - Code for argument parsing and file path handling
    - Implementation of helper functions like load_script, load_dev_texts, character_counts, and character_coverage
    - Starter code for functions (measure_bpe, measure_unigram) that train models at various vocabulary sizes, perform scoring and result recording
    - All implementation for plot and table generation
    - Code proofreading and improvements
"""

import argparse
import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

import sentencepiece as spm

from data import iter_split_files


VOCAB_SIZES = [5000, 10000, 20000, 50000]

# unk plus one piece per byte.
META_PIECES = 257

LANGUAGES = [
    ("fi", "Finnish", "#0072B2"),
    ("hu", "Hungarian", "#56B4E9"),
    ("de", "German", "#E69F00"),
    ("ru", "Russian", "#D55E00"),
    ("ar", "Arabic", "#CC79A7"),
    ("zh", "Mandarin", "#009E73"),
    ("vi", "Vietnamese", "#000000"),
]


def load_script(filename):
    """Import a sibling script by filename."""
    path = Path(__file__).resolve().parent / filename
    spec = importlib.util.spec_from_file_location(filename.replace(".", "_"), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def character_counts(input_dir):
    """Count training characters. Spaces are stored as U+2581."""
    counts = Counter()
    for path in iter_split_files(input_dir, "train"):
        for line in path.read_text(encoding="utf-8").split("\n"):
            counts.update(line.replace(" ", "\u2581"))
    return counts


def character_coverage(counts, vocab_size, margin):
    """Fraction of characters that fit under the vocabulary cap."""
    cap = vocab_size - META_PIECES - margin
    if cap < 1:
        raise ValueError(f"vocab size {vocab_size} is too small")
    items = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    if len(items) <= cap:
        return 1.0
    total = sum(counts.values())
    kept = sum(freq for _, freq in items[:cap])
    return kept / total


def bpe_merge_map(merges, vocab_size):
    """Map the first V - 256 merges to token ids."""
    return {
        pair: 256 + index
        for index, pair in enumerate(merges[: vocab_size - 256])
    }


def load_dev_texts(input_dir):
    """Read each language's cleaned dev.txt."""
    texts = {}
    for path in iter_split_files(input_dir, "dev"):
        texts[path.parent.name] = path.read_text(encoding="utf-8")
    missing = [code for code, _, _ in LANGUAGES if code not in texts]
    if missing:
        raise ValueError(f"missing dev.txt for {', '.join(missing)}")
    return texts


def write_bpe_artifacts(output_dir, vocab, merges):
    """Write bpe_vocab.txt and bpe_merges.txt for one vocabulary size."""
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "bpe_vocab.txt").open("w", encoding="utf-8", newline="\n") as handle:
        for token in vocab:
            handle.write(token.hex() + "\n")
    with (output_dir / "bpe_merges.txt").open("w", encoding="utf-8", newline="\n") as handle:
        for left, right in merges:
            handle.write(f"{vocab[left].hex()} {vocab[right].hex()}\n")


def record(rows, code, tokenizer, vocab_size, n_tokens, n_bytes):
    """Store one tokens-per-byte row and print it."""
    rows.append((code, tokenizer, vocab_size, n_tokens, n_bytes))
    print(
        f"{tokenizer} V={vocab_size} {code} {n_tokens / n_bytes:.4f}",
        file=sys.stderr,
    )


def measure_bpe(input_dir, dev_texts, encode, output_dir):
    """Train BPE once, save each vocabulary size, and score dev."""
    bpe = load_script("part1.2-bpe.py")
    raw, ends = bpe.load_corpus(input_dir)
    vocab, merges = bpe.train_bpe(raw, ends, max(VOCAB_SIZES))
    rows = []
    for vocab_size in VOCAB_SIZES:
        write_bpe_artifacts(
            Path(output_dir) / "bpe" / f"v{vocab_size}",
            vocab[:vocab_size],
            merges[: vocab_size - 256],
        )
        merge_of = bpe_merge_map(merges, vocab_size)
        for code, _, _ in LANGUAGES:
            text = dev_texts[code]
            n_bytes = len(text.encode("utf-8"))
            n_tokens = len(encode(text.encode("utf-8"), merge_of))
            record(rows, code, "bpe", vocab_size, n_tokens, n_bytes)
    return rows


def measure_unigram(input_dir, dev_texts, counts, output_dir):
    """Train unigram at each size, save the model, and score dev."""
    unigram = load_script("part1.4-unigram.py")
    rows = []
    for vocab_size in VOCAB_SIZES:
        margin = 200
        model = None
        while True:
            coverage = character_coverage(counts, vocab_size, margin)
            print(
                f"unigram V={vocab_size} coverage={coverage:.6f} margin={margin}",
                file=sys.stderr,
            )
            try:
                model_dir = Path(output_dir) / "unigram" / f"v{vocab_size}"
                model_dir.mkdir(parents=True, exist_ok=True)
                prefix = str(model_dir / "unigram")
                with open(os.devnull, "w", encoding="utf-8") as sink:
                    unigram.train_unigram(
                        input_dir,
                        prefix,
                        vocab_size,
                        character_coverage=coverage,
                        logstream=sink,
                    )
                model = spm.SentencePieceProcessor()
                model.Load(prefix + ".model")
                break
            except RuntimeError as error:
                message = str(error)
                if "vocab" not in message.lower() and "character" not in message.lower():
                    raise
                margin += max(500, margin)
                print(f"retrying with margin={margin}: {message}", file=sys.stderr)
                if margin > vocab_size // 2:
                    raise
        if model.GetPieceSize() != vocab_size:
            raise RuntimeError(
                f"unigram vocab is {model.GetPieceSize()}, expected {vocab_size}"
            )
        for code, _, _ in LANGUAGES:
            text = dev_texts[code]
            n_bytes = len(text.encode("utf-8"))
            n_tokens = len(model.Encode(text))
            record(rows, code, "unigram", vocab_size, n_tokens, n_bytes)
    return rows


def write_table(path, rows):
    """Write compression_ratios.tsv."""
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("language\ttokenizer\tvocab_size\ttokens\tbytes\ttokens_per_byte\n")
        for code, tokenizer, vocab_size, n_tokens, n_bytes in rows:
            handle.write(
                f"{code}\t{tokenizer}\t{vocab_size}\t{n_tokens}\t{n_bytes}\t"
                f"{n_tokens / n_bytes:.6f}\n"
            )


def _find_chrome():
    """Return a Chrome or Chromium binary, if one is installed."""
    names = ("google-chrome", "chromium", "chromium-browser")
    found = [shutil.which(name) for name in names]
    found.extend(
        [
            "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
            "/Applications/Chromium.app/Contents/MacOS/Chromium",
        ]
    )
    for candidate in found:
        if candidate and Path(candidate).is_file():
            return candidate
    return None


def _write_png(svg_text, path):
    """Rasterize the chart SVG to a PNG file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        svg_path = Path(tmp) / "chart.svg"
        rendered = Path(tmp) / "chart.png"
        svg_path.write_text(svg_text, encoding="utf-8")
        if shutil.which("rsvg-convert"):
            subprocess.run(
                ["rsvg-convert", "-o", str(rendered), str(svg_path)],
                check=True,
            )
        else:
            chrome = _find_chrome()
            if chrome is None:
                raise RuntimeError("writing the PNG needs rsvg-convert or Chrome")
            subprocess.run(
                [
                    chrome,
                    "--headless",
                    "--disable-gpu",
                    "--window-size=1280,700",
                    f"--screenshot={rendered.resolve()}",
                    svg_path.resolve().as_uri(),
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        path.write_bytes(rendered.read_bytes())


def write_plot(path, rows):
    """Write compression.png."""
    by_key = {
        (code, tokenizer, vocab_size): n_tokens / n_bytes
        for code, tokenizer, vocab_size, n_tokens, n_bytes in rows
    }
    y_min, y_max = 0.1, 0.7
    y_ticks = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    width, height = 1280, 700
    panel_w, panel_h = 460, 400
    panels = (("BPE", 150), ("Unigram", 740))
    top = 110

    def x_of(vocab_size, left):
        """Return the horizontal position of a vocabulary size."""
        index = VOCAB_SIZES.index(vocab_size)
        return left + panel_w * index / (len(VOCAB_SIZES) - 1)

    def y_of(ratio):
        """Return the vertical position of a tokens-per-byte ratio."""
        return top + panel_h * (y_max - ratio) / (y_max - y_min)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="640" y="42" text-anchor="middle" font-family="sans-serif" '
        'font-size="26">Compression ratio (tokens per byte) v.s. vocabulary size</text>',
        '<text x="36" y="340" text-anchor="middle" font-family="sans-serif" '
        'font-size="18" transform="rotate(-90 36 340)">tokens / byte</text>',
    ]
    for name, left in panels:
        parts.append(
            f'<text x="{left + panel_w / 2:.0f}" y="{top - 16}" text-anchor="middle" '
            f'font-family="sans-serif" font-size="22">{name}</text>'
        )
        for vocab_size in VOCAB_SIZES:
            x = x_of(vocab_size, left)
            parts.append(
                f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + panel_h}" '
                'stroke="#eeeeee"/>'
            )
        for ratio in y_ticks:
            y = y_of(ratio)
            parts.append(
                f'<line x1="{left}" y1="{y:.1f}" x2="{left + panel_w}" y2="{y:.1f}" '
                'stroke="#eeeeee"/>'
            )
            parts.append(
                f'<text x="{left - 12}" y="{y + 6:.1f}" text-anchor="end" '
                f'font-family="sans-serif" font-size="16">{ratio:.1f}</text>'
            )
        parts.append(
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + panel_h}" '
            'stroke="#222"/>'
        )
        parts.append(
            f'<line x1="{left}" y1="{top + panel_h}" x2="{left + panel_w}" '
            f'y2="{top + panel_h}" stroke="#222"/>'
        )
        for vocab_size in VOCAB_SIZES:
            x = x_of(vocab_size, left)
            label = f"{vocab_size // 1000}k"
            parts.append(
                f'<text x="{x:.1f}" y="{top + panel_h + 28}" text-anchor="middle" '
                f'font-family="sans-serif" font-size="16">{label}</text>'
            )
        for code, _, color in LANGUAGES:
            points = []
            for vocab_size in VOCAB_SIZES:
                ratio = by_key[(code, name.lower(), vocab_size)]
                points.append(f"{x_of(vocab_size, left):.1f},{y_of(ratio):.1f}")
            parts.append(
                f'<polyline fill="none" stroke="{color}" stroke-width="2" '
                f'points="{" ".join(points)}"/>'
            )
            for vocab_size in VOCAB_SIZES:
                ratio = by_key[(code, name.lower(), vocab_size)]
                parts.append(
                    f'<circle cx="{x_of(vocab_size, left):.1f}" cy="{y_of(ratio):.1f}" '
                    f'r="3.2" fill="{color}"/>'
                )
        parts.append(
            f'<text x="{left + panel_w / 2:.0f}" y="{top + panel_h + 56}" '
            'text-anchor="middle" font-family="sans-serif" font-size="18">'
            "vocabulary size</text>"
        )

    legend_y = 660
    for index, (_, name, color) in enumerate(LANGUAGES):
        x = 40 + index * 176
        parts.append(
            f'<line x1="{x}" y1="{legend_y}" x2="{x + 28}" y2="{legend_y}" '
            f'stroke="{color}" stroke-width="4"/>'
        )
        parts.append(
            f'<text x="{x + 34}" y="{legend_y + 6}" font-family="sans-serif" '
            f'font-size="18">{name}</text>'
        )
    parts.append("</svg>")
    _write_png("\n".join(parts) + "\n", path)


def main():
    """Sweep both tokenizers and write the table and plot."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    dev_texts = load_dev_texts(args.input)
    encode = load_script("part1.3-encode.py").encode

    rows = measure_bpe(args.input, dev_texts, encode, output_dir)
    rows.extend(
        measure_unigram(args.input, dev_texts, character_counts(args.input), output_dir)
    )
    write_table(output_dir / "compression_ratios.tsv", rows)
    write_plot(output_dir / "compression.png", rows)
    print(f"wrote {output_dir / 'compression.png'}", file=sys.stderr)


if __name__ == "__main__":
    main()
