"""
Train the Assignment 2 Transformer language model.

Paths in the YAML file are resolved relative to that file's directory.
The run writes artifacts/part2/config.yaml with those paths rewritten so the
same file still works after the project directory is moved.

This file was written with assistance from generative AI.
All AI-generated code was read over and checked for correctness.
"""

import argparse
import logging
from pathlib import Path

import yaml

from part1_data import load_documents
from part1_train import train
from part2_model import TransformerLanguageModel
from run_part1 import apply_runtime, choose_device, output_paths


def main():
    parser = argparse.ArgumentParser(description="Train the byte-level Transformer language model")
    parser.add_argument("--config", required=True, help="Path to the YAML configuration")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.info("process started")

    config_path = Path(args.config).resolve()
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    apply_runtime(config)
    device = choose_device(config)
    train_documents, dev_documents = load_documents(config)
    logging.info("documents train=%d dev=%d", len(train_documents), len(dev_documents))
    paths = output_paths(config, config_path)
    train(
        config,
        train_documents,
        dev_documents,
        device,
        paths,
        model_cls=TransformerLanguageModel,
    )
    logging.info("training complete")


if __name__ == "__main__":
    main()
