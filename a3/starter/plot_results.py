"""STUDENT IMPLEMENTATION: validate 65 runs, aggregate five seeds, and plot.

Require exactly layers 0..12 and seeds 0..4, without duplicates, mixed splits,
nonfinite scores, inconsistent counts, or mixed data/protocol fingerprints.
Write 13 rows with fields:
layer,split,n_seeds,mean_accuracy,std_accuracy,se_accuracy,data_sha256,protocol_sha256
Use std(ddof=1) / sqrt(5), not population SD or binomial standard error.
Plot mean dev accuracy and +/- 1 SE. Label layer 0 as embedding-module output.
"""

import argparse
import csv
import math
from pathlib import Path

import matplotlib

# Agg draws the PDF without opening a window, so this works over SSH or in a script.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from protocol import LAYERS, SEEDS, validate_sha256

# These names match evaluator.py. This script reads that file; it does not retrain.
RUN_FIELDS = [
    "layer", "seed", "split", "best_epoch", "num_correct", "num_spans",
    "accuracy", "data_sha256", "protocol_sha256",
]
LAYER_FIELDS = [
    "layer", "split", "n_seeds", "mean_accuracy", "std_accuracy", "se_accuracy",
    "data_sha256", "protocol_sha256",
]


def main():
    """Read runs.csv, write the 13-row summary, and save the layer figure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--table", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    aggregate_runs(args.input, args.table, args.output)


def aggregate_runs(input_path, table_path, output_path):
    """Validate all 65 runs, then write layers.csv and the accuracy PDF.

    Nothing is averaged until the file is known to contain each layer and
    seed once, from one split and one data and protocol fingerprint.
    """
    rows = read_runs(input_path)
    summary = summarize_runs(rows)
    write_table(table_path, summary)
    plot_layers(output_path, summary)
    return summary


def read_runs(path):
    """Load runs.csv and reject a file that cannot be compared across seeds.

    Accuracy is a property of one trained probe on the full split. Mixing
    splits, datasets, or span counts would make the mean meaningless.
    """
    with Path(path).open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != RUN_FIELDS:
            raise ValueError("runs.csv header does not match the required fields")
        rows = [parse_run(raw) for raw in reader]
    # A set comparison catches both a missing pair and a duplicated pair.
    expected = {(layer, seed) for layer in LAYERS for seed in SEEDS}
    seen = {(row["layer"], row["seed"]) for row in rows}
    if len(rows) != len(expected) or seen != expected:
        raise ValueError("runs.csv must contain layers 0..12 and seeds 0..4 exactly once")
    if len({row["split"] for row in rows}) != 1:
        raise ValueError("Mixed evaluation splits")
    if len({row["data_sha256"] for row in rows}) != 1 or len({row["protocol_sha256"] for row in rows}) != 1:
        raise ValueError("Mixed data or protocol fingerprints")
    if len({row["num_spans"] for row in rows}) != 1:
        raise ValueError("Inconsistent span counts")
    return rows


def parse_run(raw):
    """Turn one CSV row into numbers and check that its accuracy is a real count.

    accuracy must be num_correct / num_spans. A hand-edited fraction that does
    not match the integer counts is rejected.
    """
    if any(raw.get(field) in (None, "") for field in RUN_FIELDS):
        raise ValueError("Incomplete runs row")
    try:
        layer = int(raw["layer"])
        seed = int(raw["seed"])
        best_epoch = int(raw["best_epoch"])
        num_correct = int(raw["num_correct"])
        num_spans = int(raw["num_spans"])
        accuracy = float(raw["accuracy"])
    except (TypeError, ValueError) as error:
        raise ValueError("Invalid runs row") from error
    if layer not in LAYERS or seed not in SEEDS or not 1 <= best_epoch <= 10:
        raise ValueError("Invalid layer, seed, or best epoch")
    if not 0 <= num_correct <= num_spans or num_spans == 0:
        raise ValueError("Inconsistent counts")
    if not math.isfinite(accuracy) or not 0.0 <= accuracy <= 1.0:
        raise ValueError("Nonfinite or out-of-range accuracy")
    # Span-level accuracy, not an average of minibatch accuracies.
    if not math.isclose(accuracy, num_correct / num_spans, rel_tol=0, abs_tol=1e-12):
        raise ValueError("Accuracy does not match the span counts")
    validate_sha256(raw["data_sha256"])
    validate_sha256(raw["protocol_sha256"])
    return {
        "layer": layer,
        "seed": seed,
        "split": raw["split"],
        "best_epoch": best_epoch,
        "num_correct": num_correct,
        "num_spans": num_spans,
        "accuracy": accuracy,
        "data_sha256": raw["data_sha256"],
        "protocol_sha256": raw["protocol_sha256"],
    }


def summarize_runs(rows):
    """Average the five seeds of each layer and estimate their variability.

    The standard deviation divides by 4 (ddof=1), the sample formula for five
    probes. The standard error divides that by sqrt(5). Both describe variation
    across seeds, not across spans. Dividing by 5 would be the population
    formula and would make the error bars too small.
    """
    summary = []
    for layer in LAYERS:
        group = sorted((row for row in rows if row["layer"] == layer), key=lambda row: row["seed"])
        if [row["seed"] for row in group] != list(SEEDS):
            raise ValueError("Each layer must contain seeds 0..4")
        accuracies = np.asarray([row["accuracy"] for row in group], dtype=np.float64)
        if accuracies.shape != (len(SEEDS),) or not np.isfinite(accuracies).all():
            raise ValueError("Nonfinite accuracy")
        standard_deviation = float(accuracies.std(ddof=1))
        standard_error = standard_deviation / math.sqrt(len(SEEDS))
        summary.append({
            "layer": layer,
            "split": group[0]["split"],
            "n_seeds": len(SEEDS),
            "mean_accuracy": float(accuracies.mean()),
            "std_accuracy": standard_deviation,
            "se_accuracy": standard_error,
            "data_sha256": group[0]["data_sha256"],
            "protocol_sha256": group[0]["protocol_sha256"],
        })
    return summary


def write_table(path, summary):
    """Write layers.csv with accuracies kept as fractions in [0, 1].

    The plot converts to percent later. The table stays in the same units as
    runs.csv so the two files can be compared directly.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=LAYER_FIELDS)
        writer.writeheader()
        for row in summary:
            writer.writerow({
                **row,
                "mean_accuracy": format_float(row["mean_accuracy"]),
                "std_accuracy": format_float(row["std_accuracy"]),
                "se_accuracy": format_float(row["se_accuracy"]),
            })


def plot_layers(path, summary):
    """Plot mean development accuracy against layer, with a ±1 SE error bar.

    Both the mean and the standard error are multiplied by 100, so the bars
    are in percentage points. Layer 0 is the embedding output, not a block.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    layers = [row["layer"] for row in summary]
    means = [row["mean_accuracy"] * 100 for row in summary]
    errors = [row["se_accuracy"] * 100 for row in summary]
    figure, axis = plt.subplots(figsize=(9, 4.8))
    axis.errorbar(
        layers, means, yerr=errors, fmt="-o", capsize=4, linewidth=1.5, markersize=5, label="Mean ± 1 SE",
    )
    axis.set_xticks(list(LAYERS))
    axis.set_xticklabels(["0\nembedding-module output", *[str(layer) for layer in LAYERS if layer != 0]])
    axis.set_xlabel("Layer (0 = embedding-module output)")
    axis.set_ylabel(f"{summary[0]['split']} accuracy (%)")
    axis.set_title("Frozen BERT constituent-probe accuracy")
    axis.legend()
    axis.grid(True, axis="y", linewidth=0.4, alpha=0.6)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def format_float(value):
    """Format a fraction without turning a standard deviation into scientific noise.

    .17g keeps enough digits to round-trip a float64 and still writes 0 as 0.
    """
    if not math.isfinite(value) or value < 0 or value > 1:
        raise ValueError("Aggregate accuracy must be a fraction in [0, 1]")
    return format(value, ".17g")


if __name__ == "__main__":
    main()
