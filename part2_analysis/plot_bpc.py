import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


VOCAB_SIZES = [5000, 10000, 20000, 50000]
ALGORITHMS = [
    ("wordpiece", "WordPiece", "#D55E00"),
    ("bpe", "BPE", "#0072B2"),
    ("unigram", "Unigram", "#009E73"),
]
RECORDED = {
    ("wordpiece", 5000): (3.0451, 2.1103, 6752637, 6751325),
    ("wordpiece", 10000): (3.0441, 2.1446, 6642408, 6751325),
    ("wordpiece", 20000): (3.0570, 2.2792, 6276710, 6751325),
    ("wordpiece", 50000): (3.0241, 2.7151, 5212332, 6751325),
    ("bpe", 5000): (2.9777, 3.9267, 3548664, 6751325),
    ("bpe", 10000): (2.9379, 4.5049, 3051865, 6751325),
    ("bpe", 20000): (2.9322, 5.1933, 2642188, 6751325),
    ("bpe", 50000): (2.9670, 6.2864, 2208668, 6751325),
    ("unigram", 5000): (2.9103, 2.6910, 5061134, 6751325),
    ("unigram", 10000): (2.9055, 2.6933, 5048277, 6751325),
    ("unigram", 20000): (2.7914, 4.4151, 2958647, 6751325),
    ("unigram", 50000): (2.6768, 5.3457, 2343316, 6751325),
}


def load_runs(runs_dir):
    """Read dev_metrics.json from runs_dir/<algorithm>/v<size>/."""
    rows = {}
    for algorithm, _, _ in ALGORITHMS:
        for vocab_size in VOCAB_SIZES:
            path = Path(runs_dir) / algorithm / f"v{vocab_size}" / "dev_metrics.json"
            if not path.is_file():
                raise FileNotFoundError(f"missing {path}")
            metrics = json.loads(path.read_text(encoding="utf-8"))
            rows[(algorithm, vocab_size)] = (
                metrics["dev_bpc"],
                metrics["dev_nll_per_token"],
                metrics["dev_tokens_scored"],
                metrics["dev_chars"],
            )
    return rows


def write_table(path, rows):
    """Write bpc_vs_v.tsv."""
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(
            "algorithm\tvocab_size\tdev_bpc\tnll_per_token\t"
            "tokens_scored\tchars\ttokens_per_char\n"
        )
        for algorithm, _, _ in ALGORITHMS:
            for vocab_size in VOCAB_SIZES:
                bpc, nll, tokens, chars = rows[(algorithm, vocab_size)]
                handle.write(
                    f"{algorithm}\t{vocab_size}\t{bpc:.4f}\t{nll:.4f}\t"
                    f"{tokens}\t{chars}\t{tokens / chars:.6f}\n"
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
    """Write bpc_vs_v.png."""
    y_min, y_max = 2.6, 3.1
    y_ticks = [2.6, 2.7, 2.8, 2.9, 3.0, 3.1]
    width, height = 1280, 700
    left, top = 160, 110
    panel_w, panel_h = 960, 400

    def x_of(vocab_size):
        """Return the horizontal position of a vocabulary size."""
        index = VOCAB_SIZES.index(vocab_size)
        return left + panel_w * index / (len(VOCAB_SIZES) - 1)

    def y_of(bpc):
        """Return the vertical position of a bits-per-character value."""
        return top + panel_h * (y_max - bpc) / (y_max - y_min)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="640" y="42" text-anchor="middle" font-family="sans-serif" '
        'font-size="26">Dev bits per character v.s. vocabulary size</text>',
        '<text x="36" y="340" text-anchor="middle" font-family="sans-serif" '
        'font-size="24" transform="rotate(-90 36 340)">bits per character</text>',
    ]
    for vocab_size in VOCAB_SIZES:
        x = x_of(vocab_size)
        parts.append(
            f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + panel_h}" '
            'stroke="#eeeeee"/>'
        )
    for bpc in y_ticks:
        y = y_of(bpc)
        parts.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{left + panel_w}" y2="{y:.1f}" '
            'stroke="#eeeeee"/>'
        )
        parts.append(
            f'<text x="{left - 12}" y="{y + 8:.1f}" text-anchor="end" '
            f'font-family="sans-serif" font-size="22">{bpc:.1f}</text>'
        )
    parts.append(
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + panel_h}" stroke="#222"/>'
    )
    parts.append(
        f'<line x1="{left}" y1="{top + panel_h}" x2="{left + panel_w}" '
        f'y2="{top + panel_h}" stroke="#222"/>'
    )
    for vocab_size in VOCAB_SIZES:
        x = x_of(vocab_size)
        parts.append(
            f'<text x="{x:.1f}" y="{top + panel_h + 34}" text-anchor="middle" '
            f'font-family="sans-serif" font-size="22">{vocab_size // 1000}k</text>'
        )
    for algorithm, _name, color in ALGORITHMS:
        points = []
        for vocab_size in VOCAB_SIZES:
            bpc = rows[(algorithm, vocab_size)][0]
            points.append(f"{x_of(vocab_size):.1f},{y_of(bpc):.1f}")
        parts.append(
            f'<polyline fill="none" stroke="{color}" stroke-width="2" '
            f'points="{" ".join(points)}"/>'
        )
        for vocab_size in VOCAB_SIZES:
            bpc = rows[(algorithm, vocab_size)][0]
            parts.append(
                f'<circle cx="{x_of(vocab_size):.1f}" cy="{y_of(bpc):.1f}" '
                f'r="3.2" fill="{color}"/>'
            )
    parts.append(
        f'<text x="{left + panel_w / 2:.0f}" y="{top + panel_h + 68}" '
        'text-anchor="middle" font-family="sans-serif" font-size="24">'
        "vocabulary size</text>"
    )
    for algorithm, _name, _color in ALGORITHMS:
        best_size = min(VOCAB_SIZES, key=lambda size: rows[(algorithm, size)][0])
        best_bpc = rows[(algorithm, best_size)][0]
        parts.append(
            f'<circle cx="{x_of(best_size):.1f}" cy="{y_of(best_bpc):.1f}" r="8" '
            'fill="none" stroke="#e31a1c" stroke-width="2"/>'
        )
    legend_y = 660
    for index, (_algorithm, name, color) in enumerate(ALGORITHMS):
        x = 250 + index * 270
        parts.append(
            f'<line x1="{x}" y1="{legend_y}" x2="{x + 36}" y2="{legend_y}" '
            f'stroke="{color}" stroke-width="5"/>'
        )
        parts.append(
            f'<text x="{x + 44}" y="{legend_y + 8}" font-family="sans-serif" '
            f'font-size="24">{name}</text>'
        )
    parts.append("</svg>")
    _write_png("\n".join(parts) + "\n", path)


def main():
    """Write the bits-per-character table and plot."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--runs", type=str, default=None)
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_runs(args.runs) if args.runs else RECORDED
    write_table(output_dir / "bpc_vs_v.tsv", rows)
    write_plot(output_dir / "bpc_vs_v.png", rows)
    print(f"wrote {output_dir / 'bpc_vs_v.png'}", file=sys.stderr)
    for algorithm, name, _color in ALGORITHMS:
        vocab_size, (bpc, _nll, _tokens, _chars) = min(
            ((size, rows[(algorithm, size)]) for size in VOCAB_SIZES),
            key=lambda item: item[1][0],
        )
        print(f"sweet spot {name} V={vocab_size} dev_bpc={bpc:.4f}", file=sys.stderr)


if __name__ == "__main__":
    main()
