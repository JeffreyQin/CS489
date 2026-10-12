"""Score saved n-gram counts on each language's dev file.

This script only loads the count files written by part3.py and applies each
alpha at scoring time. The plotted
point at each order is the lowest perplexity among the alphas.

    uv run python part3_analysis/eval_languages.py \
        --input artifacts/part1/cleaned \
        --bpe artifacts/part1/bpe/ \
        --counts artifacts/part3/counts \
        --output artifacts/part3/analysis
"""

import argparse
import math
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from part3_artifacts import load_ngram_model
from part3_ngram_lm import (
    dev_perplexity,
    load_encoder,
    pack,
    tokenize_line,
)

DEFAULT_ALPHAS = ["0.00001", "0.0001", "0.001", "0.01", "0.1", "1.0"]
DEFAULT_LANGUAGES = ["fi", "zh"]
ORDERS = [2, 3, 4, 5]
LANGUAGE_STYLE = {
    "ar": ("Arabic", "#000000"),
    "de": ("German", "#E69F00"),
    "fi": ("Finnish", "#0072B2"),
    "hu": ("Hungarian", "#F0E442"),
    "ru": ("Russian", "#56B4E9"),
    "vi": ("Vietnamese", "#009E73"),
    "zh": ("Mandarin", "#D55E00"),
}


def packed_counts(model):
    """Turn saved string keys back into the packed tables used for scoring."""
    order = model["n"]
    counts = [None] + [{} for _ in range(order)]
    total = 0
    for length in range(1, order + 1):
        table = counts[length]
        for key, count in model["counts"][str(length)].items():
            token_ids = [int(part) for part in key.split()] if key else []
            table[pack(token_ids)] = count
            if length == 1:
                total += count
    return counts, total


def dev_lines(path):
    """Yield one cleaned dev file as lines of bytes, without the newline."""
    data = Path(path).read_bytes()
    if not data:
        return
    parts = data.split(b"\n")
    if data.endswith(b"\n"):
        parts.pop()
    yield from parts


def tokenize_dev(path, encode_line, merge_of, eos_id):
    """Tokenize one language's dev.txt and append EOS on every line."""
    return [
        tokenize_line(line, encode_line, merge_of, eos_id)
        for line in dev_lines(path)
    ]


def score_language(lines, order, counts, total, alpha, vocab_size, n_chars):
    """Return perplexity and bits per character for one alpha."""
    perplexity, missing, scored = dev_perplexity(
        lines,
        order,
        counts,
        total,
        alpha=alpha,
        vocab_size=vocab_size,
    )
    if missing is not None:
        return math.inf, math.inf, scored
    bpc = scored / n_chars * math.log2(perplexity)
    return perplexity, bpc, scored


def locate_model(counts_dir, order):
    """Find order{n}/model.json.gz under the counts directory."""
    path = Path(counts_dir) / f"order{order}" / "model.json.gz"
    if not path.is_file():
        raise FileNotFoundError(
            f"missing {path}; run part3.py with ORDER = {order} and "
            f"--output {Path(counts_dir) / f'order{order}'}"
        )
    return path


def write_table(path, rows):
    """Write every language, order, and alpha."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("language\torder\talpha\tperplexity\tbpc\ttokens\tchars\n")
        for row in rows:
            perplexity = row["perplexity"]
            bpc = row["bpc"]
            perplexity_text = "inf" if math.isinf(perplexity) else f"{perplexity:.6f}"
            bpc_text = "inf" if math.isinf(bpc) else f"{bpc:.4f}"
            handle.write(
                f"{row['language']}\t{row['order']}\t{row['alpha']}\t"
                f"{perplexity_text}\t{bpc_text}\t{row['tokens']}\t{row['chars']}\n"
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


def write_plot(path, best_rows, languages):
    """Plot the lowest perplexity at each order. The y-axis is logarithmic."""
    width, height = 1280, 700
    left, top = 160, 110
    panel_w, panel_h = 960, 400
    values = [row["perplexity"] for row in best_rows if row["perplexity"] > 0]
    log_min = math.log10(min(values))
    log_max = math.log10(max(values))
    pad = max(0.05, (log_max - log_min) * 0.12)
    log_min -= pad
    log_max += pad

    def x_of(order):
        return left + (order - 2) / 3 * panel_w

    def y_of(perplexity):
        fraction = (math.log10(perplexity) - log_min) / (log_max - log_min)
        return top + panel_h - fraction * panel_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="640" y="58" text-anchor="middle" font-family="sans-serif" '
        'font-size="26">Best token perplexity v.s. n-gram order</text>',
        f'<text x="36" y="340" text-anchor="middle" font-family="sans-serif" '
        f'font-size="24" transform="rotate(-90 36 340)">token perplexity</text>',
        f'<text x="640" y="580" text-anchor="middle" font-family="sans-serif" '
        f'font-size="24">n-gram order</text>',
    ]
    for tick in range(4):
        order = tick + 2
        x = x_of(order)
        parts.append(
            f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + panel_h}" '
            'stroke="#eeeeee"/>'
        )
        parts.append(
            f'<text x="{x:.1f}" y="{top + panel_h + 36}" text-anchor="middle" '
            f'font-family="sans-serif" font-size="22">{order}</text>'
        )
    decades = range(math.floor(log_min), math.ceil(log_max) + 1)
    for decade in decades:
        if not log_min <= decade <= log_max:
            continue
        y = y_of(10**decade)
        parts.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{left + panel_w}" y2="{y:.1f}" '
            'stroke="#eeeeee"/>'
        )
        parts.append(
            f'<text x="{left - 16}" y="{y + 8:.1f}" text-anchor="end" '
            f'font-family="sans-serif" font-size="22">{10 ** decade:g}</text>'
        )
    parts.append(
        f'<rect x="{left}" y="{top}" width="{panel_w}" height="{panel_h}" '
        'fill="none" stroke="#222"/>'
    )
    for index, language in enumerate(languages):
        name, color = LANGUAGE_STYLE.get(language, (language, "#333333"))
        series = [row for row in best_rows if row["language"] == language]
        series.sort(key=lambda row: row["order"])
        points = " ".join(
            f"{x_of(row['order']):.1f},{y_of(row['perplexity']):.1f}" for row in series
        )
        parts.append(
            f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{points}"/>'
        )
        for row in series:
            parts.append(
                f'<circle cx="{x_of(row["order"]):.1f}" cy="{y_of(row["perplexity"]):.1f}" '
                f'r="3.2" fill="{color}"/>'
            )
        legend_x = 250 + index * 270
        parts.append(
            f'<line x1="{legend_x}" y1="660" x2="{legend_x + 36}" y2="660" '
            f'stroke="{color}" stroke-width="5"/>'
        )
        parts.append(
            f'<text x="{legend_x + 48}" y="668" font-family="sans-serif" '
            f'font-size="24">{name}</text>'
        )
    parts.append("</svg>")
    _write_png("\n".join(parts), path)


def main():
    """Load one count file per order and score every alpha on each language."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--bpe", type=str, required=True)
    parser.add_argument("--counts", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--language", nargs="+", default=DEFAULT_LANGUAGES)
    parser.add_argument("--alpha", nargs="+", default=DEFAULT_ALPHAS)
    args = parser.parse_args()

    encode_line, merge_of, vocab = load_encoder(args.bpe)
    eos_id = len(vocab)
    vocab_size = eos_id + 1
    prepared = {}
    for language in args.language:
        dev_path = Path(args.input) / language / "dev.txt"
        if not dev_path.is_file():
            parser.error(f"no dev.txt for language {language}")
        prepared[language] = {
            "lines": tokenize_dev(dev_path, encode_line, merge_of, eos_id),
            "chars": len(dev_path.read_text(encoding="utf-8")),
        }
        print(
            f"tokenized {language} dev lines={len(prepared[language]['lines'])} "
            f"chars={prepared[language]['chars']}",
            file=sys.stderr,
        )

    rows = []
    for order in ORDERS:
        model_path = locate_model(args.counts, order)
        model = load_ngram_model(model_path)
        if model["n"] != order:
            raise SystemExit(f"{model_path} is order {model['n']}, expected {order}")
        if model["vocab_size"] != vocab_size:
            raise SystemExit(
                f"{model_path} has vocab_size {model['vocab_size']}, "
                f"but this BPE model uses {vocab_size}"
            )
        counts, total = packed_counts(model)
        print(f"loaded order-{order} from {model_path} train tokens={total}", file=sys.stderr)
        for language in args.language:
            lines = prepared[language]["lines"]
            n_chars = prepared[language]["chars"]
            for alpha_text in args.alpha:
                alpha = float(alpha_text)
                perplexity, bpc, scored = score_language(
                    lines, order, counts, total, alpha, vocab_size, n_chars
                )
                row = {
                    "language": language,
                    "order": order,
                    "alpha": alpha_text,
                    "perplexity": perplexity,
                    "bpc": bpc,
                    "tokens": scored,
                    "chars": n_chars,
                }
                rows.append(row)
                shown = "inf" if math.isinf(perplexity) else f"{perplexity:.6f}"
                print(f"{language} order={order} alpha={alpha_text} dev perplexity={shown}")

    output = Path(args.output)
    table_path = output / "perplexity_by_language.tsv"
    plot_path = output / "perplexity_vs_n.png"
    write_table(table_path, rows)
    best_rows = []
    for language in args.language:
        for order in ORDERS:
            group = [
                row
                for row in rows
                if row["language"] == language and row["order"] == order
            ]
            best = min(group, key=lambda row: row["perplexity"])
            best_rows.append(best)
            print(
                f"{language} order={order} best alpha={best['alpha']} "
                f"dev perplexity={best['perplexity']:.6f}",
                file=sys.stderr,
            )
    write_plot(plot_path, best_rows, args.language)
    print(f"wrote {table_path}", file=sys.stderr)
    print(f"wrote {plot_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
