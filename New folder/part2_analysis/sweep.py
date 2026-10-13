"""
This file was written with assistance from generative AI. 
All AI-generated code were read-over and improved by hand for correctness.

Uses of AI include
    - End-to-end implementation of all plotting/writing utilities, such as
      write_table, write_plot, _write_png, and _find_chrome
    - Starter code for parameter parsing and file path handling

Train several vocabulary sizes and write one bits-per-character chart
"""

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


DEFAULT_SIZES = [5000, 10000, 20000, 50000]


def run_one(python, input_dir, output_dir, vocab_size, algorithm):
    """Train one vocabulary size, or reuse a finished run."""
    run_dir = output_dir / algorithm / f"v{vocab_size}"
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
            "--algorithm",
            algorithm,
        ],
        check=True,
    )
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def write_table(path, rows):
    """Write bpc_vs_v.tsv."""
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("vocab_size\tdev_bpc\ttokens_per_char\tnll_per_token\n")
        for row in rows:
            tokens_per_char = row["dev_tokens_scored"] / row["dev_chars"]
            handle.write(
                f"{row['vocab_size']}\t{row['dev_bpc']:.6f}\t"
                f"{tokens_per_char:.6f}\t{row['dev_nll_per_token']:.6f}\n"
            )


def _find_chrome():
    """Return a Chrome or Chromium binary, if one is installed."""
    names = (
        "google-chrome",
        "chromium",
        "chromium-browser",
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/Applications/Chromium.app/Contents/MacOS/Chromium",
    )
    for name in names:
        path = shutil.which(name) or name
        if Path(path).is_file():
            return path
    return None


def _write_png(svg_text, path, width, height):
    """Rasterize the chart SVG to a PNG file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        svg_path = Path(tmp) / "chart.svg"
        rendered = Path(tmp) / "chart.png"
        svg_path.write_text(svg_text, encoding="utf-8")
        # rsvg-convert, or chrome when that is not installed.
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
                    f"--window-size={width},{height}",
                    f"--screenshot={rendered.resolve()}",
                    svg_path.resolve().as_uri(),
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        path.write_bytes(rendered.read_bytes())


def write_plot(path, rows):
    """Write bpc_vs_v.png."""
    sizes = [row["vocab_size"] for row in rows]
    bpcs = [row["dev_bpc"] for row in rows]
    y_min = min(bpcs) * 0.98
    y_max = max(bpcs) * 1.02
    if y_max == y_min:
        y_max = y_min + 1
    width, height = 720, 460
    left, top, panel_w, panel_h = 70, 60, 580, 300

    def x_of(size):
        """Return the horizontal position of a vocabulary size."""
        index = sizes.index(size)
        if len(sizes) == 1:
            return left + panel_w / 2
        return left + panel_w * index / (len(sizes) - 1)

    def y_of(bpc):
        """Return the vertical position of a bits-per-character value."""
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
    _write_png("\n".join(parts) + "\n", path, width, height)


def main():
    """Sweep vocabulary sizes and write the bits-per-character plot."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument(
        "--vocab-sizes",
        type=str,
        default=",".join(str(size) for size in DEFAULT_SIZES),
    )
    parser.add_argument("--algorithm", choices=("bpe", "unigram", "wordpiece"), default="bpe")
    args = parser.parse_args()
    sizes = [int(piece) for piece in args.vocab_sizes.split(",") if piece]
    if len(sizes) < 4:
        parser.error("provide at least four vocabulary sizes")
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for vocab_size in sizes:
        metrics = run_one(sys.executable, args.input, output_dir, vocab_size, args.algorithm)
        metrics["vocab_size"] = vocab_size
        rows.append(metrics)
        print(
            f"V={vocab_size} dev_bpc={metrics['dev_bpc']:.4f} "
            f"nll/token={metrics['dev_nll_per_token']:.4f}",
            file=sys.stderr,
        )

    write_table(output_dir / "bpc_vs_v.tsv", rows)
    write_plot(output_dir / "bpc_vs_v.png", rows)
    best = min(rows, key=lambda row: row["dev_bpc"])
    print(
        f"sweet spot V={best['vocab_size']} dev_bpc={best['dev_bpc']:.4f} "
        f"tokens/char={best['dev_tokens_scored'] / best['dev_chars']:.4f} "
        f"nll/token={best['dev_nll_per_token']:.4f}"
    )


if __name__ == "__main__":
    main()
