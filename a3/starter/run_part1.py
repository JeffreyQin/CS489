"""STUDENT IMPLEMENTATION: train independent probes for every layer and seed.

Use config.load_config, data.verify_public_manifest, your features.build_cache,
probe_model.Probe, and artifacts helpers. Train only probes with the fixed
protocol. Save best full-dev accuracy, keeping the earlier epoch on ties.
Reload each best checkpoint before reporting. Export portable config.yaml,
labels.json, manifest.json, all 65 checkpoints, and results/training.csv.
Use evaluator.py to produce final results/runs.csv.
Refuse nonempty artifact/results directories; --output-root selects a fresh
root for artifacts/part1 and results while reusing the input data and cache.
"""

import argparse
import csv
import platform
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

from artifacts import initialize_artifacts, load_checkpoint, save_checkpoint, validate_artifacts
from config import configure_runtime, load_config
from data import load_labels, verify_public_manifest
from features import build_cache
from probe_model import Probe, evaluate_probe, feature_batches
from protocol import LAYERS, PROTOCOL, SEEDS

# One row per epoch, not per selected checkpoint. 13 layers x 5 seeds x 10 epochs.
TRAINING_FIELDS = [
    "layer", "seed", "epoch", "train_loss", "train_num_spans",
    "dev_loss", "dev_num_correct", "dev_num_spans", "dev_accuracy",
]


def main():
    """Parse the CLI and train every layer and seed."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-root", type=Path,
                        help="Fresh run root relative to the current directory; reuse input data and cache")
    args = parser.parse_args()
    run(args.config, args.output_root)


def run(config_path, output_root=None):
    """Train 65 independent probes and write their checkpoints and epoch log.

    BERT stays frozen. Each run gets its own probe and optimizer, reads one
    cached layer, and never continues from another seed or layer. An incomplete
    run is not resumed: the artifact and results directories must start empty.
    """
    config, paths = load_config(config_path)
    # --output-root relocates artifacts and results only. Data and cache stay put.
    paths = apply_output_root(paths, output_root)
    ensure_separated(paths)
    require_fresh_directory(paths["artifacts"], "Artifact")
    require_fresh_directory(paths["results"], "Results")
    # Sets thread count and deterministic mode before any CUDA work.
    device = configure_runtime(config["runtime"])
    labels = load_labels(paths["labels"])
    # Confirms train.jsonl, dev.jsonl, and labels.json match the published hashes.
    pins = verify_public_manifest(paths["data_manifest"], paths)
    runtime = config["runtime"]
    print(f"Preparing train features on {device}", flush=True)
    # Train and dev cannot share one cache directory: their data hashes differ.
    train_cache = build_cache(
        paths["train"], labels, paths["cache"] / "train", device=str(device),
        batch_size=runtime["feature_batch_size"], local_files_only=runtime["local_files_only"],
    )
    print(f"Preparing dev features on {device}", flush=True)
    dev_cache = build_cache(
        paths["dev"], labels, paths["cache"] / "dev", device=str(device),
        batch_size=runtime["feature_batch_size"], local_files_only=runtime["local_files_only"],
    )
    # Writes config.yaml, labels.json, and an empty manifest into artifacts/part1.
    initialize_artifacts(paths["artifacts"], config, paths, pins, runtime_record(device, runtime))
    paths["results"].mkdir(parents=True, exist_ok=True)
    training_path = paths["results"] / "training.csv"
    with training_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=TRAINING_FIELDS)
        writer.writeheader()
        for layer in LAYERS:
            # All five seeds of this layer read the same frozen matrix.
            train_features = train_cache.layer(layer)
            dev_features = dev_cache.layer(layer)
            for seed in SEEDS:
                print(f"Training layer {layer} seed {seed}", flush=True)
                probe, best_epoch, best_metrics, history = train_probe(
                    train_features, train_cache.targets, dev_features, dev_cache.targets,
                    num_labels=len(labels), seed=seed, device=device,
                    batch_size=PROTOCOL["batch_size"], eval_batch_size=runtime["eval_batch_size"],
                )
                entry = save_checkpoint(
                    paths["artifacts"], probe, layer=layer, seed=seed,
                    best_epoch=best_epoch, dev_metrics=best_metrics,
                )
                # Load the file just written, rather than trusting the in-memory weights.
                reloaded = load_checkpoint(paths["artifacts"], entry, len(labels), device=device)
                confirmed = evaluate_probe(
                    reloaded, dev_features, dev_cache.targets, device=device, batch_size=runtime["eval_batch_size"],
                )
                if confirmed["num_correct"] != best_metrics["num_correct"] or confirmed["num_spans"] != best_metrics["num_spans"]:
                    raise RuntimeError("Reloaded checkpoint does not reproduce the selected development score")
                # The CSV keeps every epoch. The checkpoint keeps only the selected one.
                writer.writerows(history_rows(layer, seed, history))
                stream.flush()
                del probe, reloaded
            # Drop this layer's memory map before opening the next one.
            release_layer(train_cache, layer)
            release_layer(dev_cache, layer)
    # Fails unless all 65 checkpoints are present and match the manifest hashes.
    validate_artifacts(paths["artifacts"], labels)
    print(f"Saved 65 checkpoints to {paths['artifacts']}", flush=True)
    print(f"Saved epoch log to {training_path}", flush=True)
    print("Run evaluator.py to write results/runs.csv", flush=True)


def runtime_record(device, runtime):
    """Record the device and library versions next to the checkpoints.

    Small accuracy differences across machines are expected. This record says
    which machine produced the run; it is not itself an accuracy result.
    """
    return {
        "device": str(device),
        "feature_batch_size": runtime["feature_batch_size"],
        "eval_batch_size": runtime["eval_batch_size"],
        "num_threads": runtime["num_threads"],
        "deterministic": runtime["deterministic"],
        "local_files_only": runtime["local_files_only"],
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
    }


def apply_output_root(paths, output_root):
    """Point artifacts and results at a fresh directory when --output-root is set.

    A relative root is resolved from the current working directory, not from
    the YAML file. The data files and feature cache keep their configured paths.
    """
    paths = dict(paths)
    if output_root is None:
        return paths
    root = Path(output_root)
    if not root.is_absolute():
        root = Path.cwd() / root
    root = root.resolve()
    paths["artifacts"] = root / "artifacts" / "part1"
    paths["results"] = root / "results"
    return paths


def ensure_separated(paths):
    """Refuse a layout where an output directory could overwrite an input.

    Cache, artifacts, and results must be separate directories, and none of
    them may contain the data files or sit inside those files' directories.
    """
    watched = ("artifacts", "results", "cache")
    for index, left_name in enumerate(watched):
        for right_name in watched[index + 1:]:
            if _nested(paths[left_name], paths[right_name]) or _nested(paths[right_name], paths[left_name]):
                raise ValueError(f"{left_name} and {right_name} paths overlap")
    inputs = ("train", "dev", "labels", "data_manifest")
    for input_name in inputs:
        for output_name in watched:
            if _nested(paths[input_name], paths[output_name]) or _nested(paths[output_name], paths[input_name]):
                raise ValueError(f"{input_name} and {output_name} paths overlap")


def require_fresh_directory(path, label):
    """Allow a missing or empty directory, and reject one that already has files.

    Training does not resume. A nonempty directory would mix a new run with
    checkpoints or rows from an earlier one.
    """
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise FileExistsError(f"{label} directory must be empty; choose a fresh run directory")


def train_probe(train_features, train_targets, dev_features, dev_targets, *, num_labels, seed, device,
                batch_size, eval_batch_size):
    """Train one fresh probe and return the weights from its best dev epoch.

    The seed controls both the initial weights and the shuffle order. Dev
    accuracy is measured on the whole dev split after every epoch. A tie keeps
    the earlier epoch because the comparison is strict greater-than.
    """
    # Seed before constructing the probe, or the initial weights ignore the seed.
    reset_random_state(seed)
    probe = Probe(num_labels).to(device)
    # A new optimizer too: Adam's moment buffers must not carry over from another run.
    optimizer = torch.optim.Adam(
        probe.parameters(),
        lr=PROTOCOL["learning_rate"],
        betas=tuple(PROTOCOL["adam_betas"]),
        eps=PROTOCOL["adam_eps"],
        weight_decay=PROTOCOL["weight_decay"],
    )
    best_epoch = None
    best_metrics = None
    best_state = None
    history = []
    for epoch in range(1, PROTOCOL["epochs"] + 1):
        train_loss, train_count = train_epoch(probe, optimizer, train_features, train_targets, batch_size, device)
        metrics = evaluate_probe(probe, dev_features, dev_targets, device=device, batch_size=eval_batch_size)
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_num_spans": train_count,
            "dev_loss": metrics["loss"],
            "dev_num_correct": metrics["num_correct"],
            "dev_num_spans": metrics["num_spans"],
            "dev_accuracy": metrics["accuracy"],
        })
        # Strict ">" keeps epoch 2 when epoch 3 has the same accuracy.
        if best_metrics is None or metrics["accuracy"] > best_metrics["accuracy"]:
            best_epoch = epoch
            best_metrics = dict(metrics)
            # Copy to CPU now. Later epochs would otherwise overwrite these tensors.
            best_state = {name: value.detach().cpu().clone() for name, value in probe.state_dict().items()}
        print(
            f"  epoch {epoch}: train_loss={train_loss:.4f} dev_accuracy={metrics['accuracy']:.4f}",
            flush=True,
        )
    # Put the selected epoch back. The probe currently holds the last epoch.
    probe.load_state_dict(best_state)
    restored = evaluate_probe(probe, dev_features, dev_targets, device=device, batch_size=eval_batch_size)
    if restored["num_correct"] != best_metrics["num_correct"] or restored["num_spans"] != best_metrics["num_spans"]:
        raise RuntimeError("Restored probe does not reproduce the selected development score")
    return probe, best_epoch, restored, history


def train_epoch(probe, optimizer, features, targets, batch_size, device):
    """Update the probe once on every training span, in a seeded random order.

    The returned loss is the mean cross-entropy per span, matching the dev
    loss from evaluate_probe. The gradient uses that same per-example mean
    inside each minibatch.
    """
    probe.train()
    # permutation consumes the NumPy stream that reset_random_state started.
    order = np.random.permutation(len(targets))
    total_loss = 0.0
    seen = 0
    for inputs, labels in feature_batches(features, targets, batch_size, device, order):
        optimizer.zero_grad(set_to_none=True)
        logits = probe(inputs)
        if not torch.isfinite(logits).all():
            raise ValueError("Nonfinite probe logits")
        # Sum, then divide by the batch length, so each span contributes equally.
        loss = nn.functional.cross_entropy(logits, labels, reduction="sum")
        (loss / len(labels)).backward()
        optimizer.step()
        total_loss += float(loss.item())
        seen += len(labels)
    if seen != len(targets):
        raise RuntimeError("Training epoch did not cover every span")
    return total_loss / seen, seen


def history_rows(layer, seed, history):
    """Attach the layer and seed to each epoch record for training.csv."""
    return [
        {
            "layer": layer,
            "seed": seed,
            "epoch": record["epoch"],
            "train_loss": record["train_loss"],
            "train_num_spans": record["train_num_spans"],
            "dev_loss": record["dev_loss"],
            "dev_num_correct": record["dev_num_correct"],
            "dev_num_spans": record["dev_num_spans"],
            "dev_accuracy": record["dev_accuracy"],
        }
        for record in history
    ]


def reset_random_state(seed):
    """Restart Python, NumPy, and PyTorch RNGs so this run is reproducible.

    Python covers library code that draws from the standard library. NumPy
    covers the span shuffle. PyTorch covers the probe's initial weights.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def release_layer(cache, layer):
    """Forget one layer's memory map after its five seeds have finished.

    The file stays on disk. Dropping the Python reference lets the operating
    system reclaim the pages before the next layer is read.
    """
    stored = getattr(cache, "_layers", None)
    if isinstance(stored, dict):
        stored.pop(layer, None)


def _nested(path, parent):
    """Return whether path is parent or a file or directory inside parent."""
    try:
        Path(path).resolve().relative_to(Path(parent).resolve())
    except ValueError:
        return False
    return True


if __name__ == "__main__":
    main()
