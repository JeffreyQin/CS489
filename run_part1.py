"""
Train the Assignment 2 LSTM language model.

Paths in the YAML file are resolved relative to that file's directory.
The run writes artifacts/part1/config.yaml with those paths rewritten so the
same file still works after the project directory is moved.

This file was written with assistance from generative AI.
All AI-generated code was read over and checked for correctness.
"""

import argparse
import logging
import os
from pathlib import Path

import torch
import yaml

from part1_data import load_documents
from part1_train import train



def output_paths(config, config_path):

    def _resolve_path(value):
        return (config_path.parent / value).resolve()

    output = config["output"]
    return {
        "artifact_dir": _resolve_path(output["artifact_dir"]),
        "metrics_file": _resolve_path(output["metrics_file"]),
        "summary_file": _resolve_path(output["summary_file"]),
        "plot_file": _resolve_path(output["plot_file"]),
        "wandb_dir": _resolve_path(config["wandb"]["dir"]),
    }


def apply_runtime(config):
    runtime = config["runtime"]
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = str(runtime["cublas_workspace_config"])
    torch.set_num_threads(int(runtime["num_threads"]))
    torch.backends.cudnn.benchmark = bool(runtime["cudnn_benchmark"])
    torch.backends.cudnn.deterministic = bool(runtime["cudnn_deterministic"])
    torch.backends.cuda.matmul.allow_tf32 = bool(runtime["allow_tf32"])
    torch.backends.cudnn.allow_tf32 = bool(runtime["allow_tf32"])
    torch.use_deterministic_algorithms(bool(runtime["deterministic_algorithms"]))


def choose_device(config):
    requested = config["runtime"]["device"]
    if requested == "cuda" and torch.cuda.is_available():
        logging.info("CUDA device available")
        return torch.device("cuda")
    if requested == "cuda":
        logging.info("CUDA device NOT available")
        return torch.device("cpu")
    logging.info("using device %s", requested)
    return torch.device(requested)


def main():
    parser = argparse.ArgumentParser(description="Train LSTM model")
    parser.add_argument("--config", required=True, help="Path to YAML configuration")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.info("process started")

    # load config
    config_path = Path(args.config).resolve()
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    # set runtime environment
    apply_runtime(config)
    device = choose_device(config)

    # load documents
    train_documents, dev_documents = load_documents(config)
    logging.info("documents train=%d dev=%d", len(train_documents), len(dev_documents))

    paths = output_paths(config, config_path)
    train(config, train_documents, dev_documents, device, paths)
    logging.info("training complete")


if __name__ == "__main__":
    main()
