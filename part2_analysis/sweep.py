"""Sweep bits-per-character against BPE vocabulary size.

Runs part2.py at several vocabulary sizes with the fixed Transformer, then
writes bpc_vs_v.svg. Each run needs a GPU and takes roughly 15 to 30 minutes.
The vocabulary sizes match the Part-1 compression sweep so the two curves
can be compared.

    uv run python part2_analysis/sweep.py \
        --input artifacts/part1/cleaned \
        --output artifacts/part2/sweep/ \
        --vocab-sizes 5000,10000,20000,50000
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


DEFAULT_SIZES = [5000, 10000, 20000, 50000]


def run_one(python, input_dir, output_dir, vocab_size):
    run_dir = output_dir / f"v{vocab_size}"
    metrics_path = run_dir / "dev_metrics.json"
    if metrics_path.is_file():
        print(f"reusing {metrics_path}", file=sys.stderr)
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    subprocess.run(
        [
            python,
            str(Path(__file__).resolve().parent.parent / "part2.py"),
            "--input",
            str(input_dir),
            "--output",
            str(run_dir),
            "--vocab-size",
            str(vocab_size),
        ],
        check=True,
    )
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def write_table(path, rows):
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("vocab_size\tdev_bpc\ttokens_per_char\tnll_per_token\n")
        for row in rows:
            tokens_per_char = row["dev_tokens_scored"] / row["dev_chars"]
            handle.write(
                f"{row['vocab_size']}\t{row['dev_bpc']:.6f}\t"
                f"{tokens_per_char:.6f}\t{row['dev_nll_per_token']:.6f}\n"
            )


def write_plot(path, rows):
    sizes = [row["vocab_size"] for row in rows]
    bpcs = [row["dev_bpc"] for row in rows]
    y_min = min(bpcs) * 0.98
    y_max = max(bpcs) * 1.02
    if y_max == y_min:
        y_max = y_min + 1
    width, height = 720, 460
    left, top, panel_w, panel_h = 70, 60, 580, 300

    def x_of(size):
        index = sizes.index(size)
        if len(sizes) == 1:
            return left + panel_w / 2
        return left + panel_w * index / (len(sizes) - 1)

    def y_of(bpc):
        return top + panel_h * (y_max - bpc) / (y_max - y_min)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="360" y="28" text-anchor="middle" font-family="sans-serif" '
        'font-size="16">Dev bits per character vs vocabulary size</text>',
        '<text x="18" y="220" text-anchor="middle" font-family="sans-serif" '
        'font-size="12" transform="rotate(-90 18 220)">bits per character</text>',
    ]
    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + panel_h}" stroke="#222"/>')
    parts.append(
        f'<line x1="{left}" y1="{top + panel_h}" x2="{left + panel_w}" '
        f'y2="{top + panel_h}" stroke="#222"/>'
    )
    for fraction in (0, 0.5, 1):
        value = y_min + (y_max - y_min) * fraction
        y = y_of(value)
        parts.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{left + panel_w}" y2="{y:.1f}" stroke="#eeeeee"/>'
        )
        parts.append(
            f'<text x="{left - 8}" y="{y + 4:.1f}" text-anchor="end" '
            f'font-family="sans-serif" font-size="11">{value:.3f}</text>'
        )
    points = []
    for size, bpc in zip(sizes, bpcs):
        x, y = x_of(size), y_of(bpc)
        points.append(f"{x:.1f},{y:.1f}")
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="#0072B2"/>')
        parts.append(
            f'<text x="{x:.1f}" y="{top + panel_h + 20}" text-anchor="middle" '
            f'font-family="sans-serif" font-size="12">{size // 1000}k</text>'
        )
    parts.append(
        f'<polyline fill="none" stroke="#0072B2" stroke-width="2" points="{" ".join(points)}"/>'
    )
    parts.append(
        f'<text x="{left + panel_w / 2:.0f}" y="{top + panel_h + 44}" '
        'text-anchor="middle" font-family="sans-serif" font-size="12">vocabulary size</text>'
    )
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument(
        "--vocab-sizes",
        type=str,
        default=",".join(str(size) for size in DEFAULT_SIZES),
    )
    args = parser.parse_args()
    sizes = [int(piece) for piece in args.vocab_sizes.split(",") if piece]
    if len(sizes) < 4:
        parser.error("provide at least four vocabulary sizes")
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for vocab_size in sizes:
        metrics = run_one(sys.executable, args.input, output_dir, vocab_size)
        metrics["vocab_size"] = vocab_size
        rows.append(metrics)
        print(
            f"V={vocab_size} dev_bpc={metrics['dev_bpc']:.4f} "
            f"nll/token={metrics['dev_nll_per_token']:.4f}",
            file=sys.stderr,
        )

    write_table(output_dir / "bpc_vs_v.tsv", rows)
    write_plot(output_dir / "bpc_vs_v.svg", rows)
    best = min(rows, key=lambda row: row["dev_bpc"])
    print(
        f"sweet spot V={best['vocab_size']} dev_bpc={best['dev_bpc']:.4f} "
        f"tokens/char={best['dev_tokens_scored'] / best['dev_chars']:.4f} "
        f"nll/token={best['dev_nll_per_token']:.4f}"
    )


if __name__ == "__main__":
    main()
