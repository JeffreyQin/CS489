"""
Document-local UTF-8 byte blocks for the fixed 256-byte protocol

This file was written with assistance from generative AI.
All AI-generated code was read over and checked for correctness.
"""

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset

from evaluator import BLOCK_LENGTH, BOS_ID, PAD_ID


class ByteBlockDataset(Dataset):
    """
    Each item is one non-overlapping byte block from a single document.
    """

    def __init__(self, documents):
        self.blocks = []
        for document in documents:
            encoded = document.encode("utf-8")
            for start in range(0, len(encoded), BLOCK_LENGTH):
                self.blocks.append(encoded[start : start + BLOCK_LENGTH])

    def __len__(self):
        return len(self.blocks)

    def __getitem__(self, index):
        return self.blocks[index]


def collate_blocks(blocks):
    """
    For block b0 ... b{m-1}:
        input: [BOS, b0, ..., b{m-2}]
        target: [b0, b1, ..., b{m-1}]
    Pad positions use PAD_ID (ignored by training loss)
    """
    max_len = max(len(block) for block in blocks)
    batch_size = len(blocks)
    input_ids = torch.full((batch_size, max_len), PAD_ID, dtype=torch.long)
    targets = torch.full((batch_size, max_len), PAD_ID, dtype=torch.long)
    for row, block in enumerate(blocks):
        byte_ids = torch.from_numpy(np.frombuffer(block, dtype=np.uint8).astype(np.int64))
        length = len(block)
        input_ids[row, 0] = BOS_ID
        if length > 1:
            input_ids[row, 1:length] = byte_ids[:-1]
        targets[row, :length] = byte_ids
    return input_ids, targets


def make_loader(dataset, config, device):
    """Minibatch loader. Shuffle is reapplied by DataLoader at each epoch."""
    training = config["training"]
    runtime = config["runtime"]
    kwargs = {
        "batch_size": int(training["batch_size"]),
        "shuffle": bool(training["shuffle"]),
        "drop_last": bool(training["drop_last"]),
        "num_workers": int(runtime["num_workers"]),
        "pin_memory": bool(runtime["pin_memory"]) and device.type == "cuda",
        "collate_fn": collate_blocks,
    }
    if training["shuffle"]:
        generator = torch.Generator()
        generator.manual_seed(int(config["seed"]))
        kwargs["generator"] = generator
    return DataLoader(dataset, **kwargs)


def load_documents(config):
    """
    Load train and dev document strings from the dataset entry in the YAML config
    """
    data_cfg = config["data"]
    loaded = load_dataset(data_cfg["dataset_id"], revision=data_cfg["revision"])
    field = data_cfg["text_field"]
    train_documents = list(loaded[data_cfg["train_split"]][field])
    dev_documents = list(loaded[data_cfg["dev_split"]][field])
    return train_documents, dev_documents
