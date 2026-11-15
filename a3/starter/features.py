"""STUDENT IMPLEMENTATION: frozen BERT extraction and reusable span caches.

Implement build_cache with the signature below and this returned interface:
cache.layer(k) is a read-only float32 (num_spans, 768) memory map;
cache.targets is an int64 label-ID memory map; cache.metadata binds data/protocol.
Use full sentences, word_ids(), all 13 hidden states, right padding, masks,
no gradients, and WordPiece mean pooling. Reuse disk shards across seeds.
"""

import gc
import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch

from config import resolve_device
from data import inspect_data, iter_sentences, wordpiece_bounds
from protocol import (
    LAYERS,
    MODEL_ID,
    PROTOCOL,
    REVISION,
    protocol_sha256,
    read_json,
    sha256_file,
    write_json,
)

# Bump this when the pooling rule changes. A new version will not reuse an old cache.
CACHE_VERSION = 1
HIDDEN_SIZE = PROTOCOL["hidden_size"]
MAX_LENGTH = PROTOCOL["max_length"]
# One matrix per BERT layer. Layer 0 is the embedding output, not the first block.
LAYER_FILES = tuple(f"layer_{layer:02d}.npy" for layer in LAYERS)
CACHE_FILES = ("metadata.json", "targets.npy", "rows.jsonl", *LAYER_FILES)


class FeatureCache:
    """Read-only view of one split's pooled span features.

    The arrays stay on disk and are memory-mapped. Training then loads one
    minibatch at a time instead of the whole multi-gigabyte cache.
    """

    def __init__(self, directory, metadata):
        """Open the label vector and remember where the layer matrices live.

        metadata["identity"] is the fingerprint written with the matrices.
        The caller has already checked that it matches this data and protocol.
        """
        self.directory = Path(directory)
        self.metadata = metadata
        identity = metadata["identity"]
        num_spans = identity["sizes"]["num_spans"]
        hidden_size = identity["sizes"]["hidden_size"]
        # mmap_mode "r" refuses writes, so a training loop cannot corrupt the cache.
        self.targets = np.load(self.directory / "targets.npy", mmap_mode="r", allow_pickle=False)
        if self.targets.dtype != np.int64 or self.targets.shape != (num_spans,) or self.targets.flags.writeable:
            raise ValueError("targets.npy must be a read-only int64 vector of label IDs")
        self._layers = {}
        self._num_spans = num_spans
        self._hidden_size = hidden_size

    def layer(self, index):
        """Return layer index as a read-only float32 matrix of shape (num_spans, 768).

        Index 0 is the embedding-module output. Indexes 1 through 12 are the
        outputs of the twelve transformer blocks. The same row order is used
        for every layer and for targets.
        """
        if type(index) is not int or index not in LAYERS:
            raise ValueError("layer must be an integer in 0..12")
        cached = self._layers.get(index)
        if cached is None:
            array = np.load(self.directory / f"layer_{index:02d}.npy", mmap_mode="r", allow_pickle=False)
            if (
                array.dtype != np.float32
                or array.shape != (self._num_spans, self._hidden_size)
                or array.flags.writeable
            ):
                raise ValueError(f"layer_{index:02d}.npy must be a read-only float32 span matrix")
            self._layers[index] = array
            cached = array
        return cached


def span_representation(token_states, bounds, start, end):
    """Mean-pool every WordPiece inside one gold constituent.

    token_states is one layer of one sentence, shape (sequence, 768), with
    [CLS] at index 0. bounds come from wordpiece_bounds and already skip
    [CLS] and [SEP]. start and end are half-open word indexes, so the last
    included word is end - 1.

    The mean is over pieces, not over words. A word split into three pieces
    contributes three vectors. Averaging inside each word first, then
    averaging those word vectors, would give a different result.
    """
    # Words in a span are contiguous, so one slice covers every piece in it.
    piece_start = bounds[start][0]
    piece_end = bounds[end - 1][1]
    if piece_end <= piece_start:
        raise ValueError("Span aligned to no WordPieces")
    pieces = token_states[piece_start:piece_end]
    # Move to float32 on CPU before the mean so every layer is stored the same way.
    pooled = pieces.detach().to(dtype=torch.float32, device="cpu").mean(dim=0)
    if pooled.shape != (HIDDEN_SIZE,) or not torch.isfinite(pooled).all():
        raise ValueError("Nonfinite or ill-shaped pooled features")
    return pooled.numpy()


def build_cache(data_path, labels, cache_dir, *, device="cpu", batch_size=8,
                local_files_only=False, tokenizer=None, encoder=None):
    """Build or reuse the pooled span cache for one JSONL split.

    Returns an object with layer(k), targets, and metadata. The same cache is
    shared by all five seeds. tokenizer and encoder are only a testing seam;
    a real run loads the pinned bert-base-cased checkpoint.
    """
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if type(local_files_only) is not bool:
        raise ValueError("local_files_only must be a boolean")
    if (tokenizer is None) != (encoder is None):
        raise ValueError("tokenizer and encoder must be supplied together")
    labels = _validate_labels(labels)
    data_path = Path(data_path)
    cache_dir = Path(cache_dir)
    if cache_dir.exists() and not cache_dir.is_dir():
        raise ValueError("cache_dir must be a directory")
    # The fingerprint decides whether an on-disk cache is still valid.
    identity = _identity(data_path, labels)
    existing = _existing_cache(cache_dir, identity)
    if existing is not None:
        return existing

    resolved_device = resolve_device(device)
    if tokenizer is None:
        tokenizer, encoder = _load_bert(local_files_only)
    # Padding must be on the right so wordpiece_bounds can find the real tokens.
    if getattr(tokenizer, "padding_side", "right") != "right":
        tokenizer.padding_side = "right"
    encoder = _freeze_encoder(encoder, resolved_device)
    _write_cache(cache_dir, data_path, labels, identity, tokenizer, encoder, resolved_device, batch_size)
    cache = FeatureCache(cache_dir, read_json(cache_dir / "metadata.json"))
    # Guard against a file that changed while the cache was being written.
    if cache.metadata["identity"] != identity:
        raise ValueError("Feature cache fingerprint mismatch")
    if sha256_file(data_path) != identity["data_sha256"]:
        raise ValueError("Dataset changed during feature extraction")
    return cache


def _validate_labels(labels):
    """Require the course label order: sorted, unique, nonempty strings.

    Probe output column i is labels[i]. A different order would train a
    different classifier, so this list is part of the cache fingerprint.
    """
    if (
        not isinstance(labels, list)
        or not labels
        or any(not isinstance(label, str) or not label for label in labels)
        or labels != sorted(set(labels))
    ):
        raise ValueError("labels must be a nonempty sorted list of unique strings")
    return labels


def _identity(data_path, labels):
    """Fingerprint the split, the label list, and the extraction procedure.

    A later run reuses the cache only when every field still matches. Changing
    the data file or the pooling protocol must produce a new directory.
    """
    info = inspect_data(data_path, labels)
    return {
        "cache_version": CACHE_VERSION,
        "data_sha256": info["data_sha256"],
        "dtype": PROTOCOL["feature_dtype"],
        "labels": list(labels),
        "protocol_sha256": protocol_sha256(labels),
        "sizes": {
            "hidden_size": HIDDEN_SIZE,
            "num_layers": len(LAYERS),
            "num_sentences": info["num_sentences"],
            "num_spans": info["num_spans"],
        },
    }


def _existing_cache(cache_dir, identity):
    """Return a finished cache, or None when the directory is absent or empty.

    A partial or mismatched directory is an error. The caller must pick a new
    directory rather than deleting or ignoring the old files.
    """
    if not cache_dir.exists():
        return None
    entries = [path for path in cache_dir.iterdir()]
    if not entries:
        return None
    names = {path.name for path in entries}
    if names != set(CACHE_FILES):
        raise ValueError("Incomplete or unexpected feature cache; use a new cache directory")
    metadata = read_json(cache_dir / "metadata.json")
    # Exact equality covers the data hash, protocol hash, labels, and sizes.
    if set(metadata) != {"identity"} or metadata["identity"] != identity:
        raise ValueError("Feature cache fingerprint mismatch; use a new cache directory")
    cache = FeatureCache(cache_dir, metadata)
    with (cache_dir / "rows.jsonl").open(encoding="utf-8") as stream:
        row_count = sum(1 for line in stream if line.strip())
    if row_count != identity["sizes"]["num_spans"]:
        raise ValueError("rows.jsonl does not match the cache fingerprint")
    return cache


def _load_bert(local_files_only):
    """Load the pinned fast tokenizer and bert-base-cased weights.

    The revision is fixed by the course protocol. There is no fallback to
    another checkpoint or to a randomly initialized encoder.
    """
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=REVISION, use_fast=True, local_files_only=local_files_only,
    )
    if not tokenizer.is_fast:
        raise ValueError("The pinned BERT tokenizer must be the fast tokenizer")
    tokenizer.padding_side = "right"
    encoder = AutoModel.from_pretrained(MODEL_ID, revision=REVISION, local_files_only=local_files_only)
    return tokenizer, encoder


def _freeze_encoder(encoder, device):
    """Run BERT as a fixed feature extractor.

    eval() turns off dropout. requires_grad_(False) keeps the backward pass,
    which only the probe should receive, from touching BERT.
    """
    if isinstance(encoder, torch.nn.Module):
        encoder.to(device)
        encoder.eval()
        for parameter in encoder.parameters():
            parameter.requires_grad_(False)
    return encoder


def _write_cache(cache_dir, data_path, labels, identity, tokenizer, encoder, device, batch_size):
    """Write the cache in a temporary directory, then publish it with one rename.

    A crash mid-write leaves the temporary directory, not a cache that the
    next run could mistake for a finished one. The rename is the commit.
    """
    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".part1-cache-", dir=cache_dir.parent))
    try:
        _populate_cache(temporary, data_path, labels, identity, tokenizer, encoder, device, batch_size)
        # Close the memory maps before renaming their files.
        gc.collect()
        if cache_dir.exists():
            cache_dir.rmdir()
        os.rename(temporary, cache_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _populate_cache(directory, data_path, labels, identity, tokenizer, encoder, device, batch_size):
    """Fill one matrix per layer, in sentence and span order.

    Row i of every layer matrix is the same constituent as targets[i].
    Only the pooled 768-vectors are stored, not the token states.
    """
    num_spans = identity["sizes"]["num_spans"]
    # Open the output files up front so each batch can append its rows.
    layers = [
        np.lib.format.open_memmap(
            directory / f"layer_{layer:02d}.npy", mode="w+", dtype=np.float32, shape=(num_spans, HIDDEN_SIZE),
        )
        for layer in LAYERS
    ]
    targets = np.lib.format.open_memmap(directory / "targets.npy", mode="w+", dtype=np.int64, shape=(num_spans,))
    # The integer target is the column the probe should predict for that label.
    label_ids = {label: index for index, label in enumerate(labels)}
    cursor = 0
    pending = []
    with (directory / "rows.jsonl").open("w", encoding="utf-8") as rows:
        for sentence in iter_sentences(data_path, labels):
            pending.append(sentence)
            if len(pending) == batch_size:
                cursor = _encode_batch(
                    pending, tokenizer, encoder, device, layers, targets, rows, label_ids, cursor,
                )
                pending = []
        if pending:
            cursor = _encode_batch(
                pending, tokenizer, encoder, device, layers, targets, rows, label_ids, cursor,
            )
    if cursor != num_spans:
        raise ValueError("Extracted span count does not match the dataset")
    if sha256_file(data_path) != identity["data_sha256"]:
        raise ValueError("Dataset changed during feature extraction")
    for array in (*layers, targets):
        array.flush()
    del layers, targets
    write_json(directory / "metadata.json", {"identity": identity})


def _encode_batch(sentences, tokenizer, encoder, device, layers, targets, rows, label_ids, cursor):
    """Encode a sentence batch and write one pooled vector per constituent.

    The whole sentence is encoded, so each piece's hidden state already
    contains context from outside the span. The span vector is then just a
    mean of the pieces inside the gold boundaries.
    """
    encoding = tokenizer(
        [sentence["tokens"] for sentence in sentences],
        is_split_into_words=True,
        add_special_tokens=True,
        padding=True,
        # Never drop pieces. A sentence that does not fit is rejected below.
        truncation=False,
        return_attention_mask=True,
        return_tensors="pt",
    )
    if not hasattr(encoding, "word_ids"):
        raise ValueError("Tokenizer must return word_ids() alignments")
    input_ids = _as_long_tensor(encoding["input_ids"])
    attention_mask = _as_long_tensor(encoding["attention_mask"])
    if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape or len(input_ids) != len(sentences):
        raise ValueError("Tokenizer must return a rectangular batch of input ids and attention masks")
    word_ids_batch = []
    masks = []
    for index, sentence in enumerate(sentences):
        word_ids = list(encoding.word_ids(index))
        mask = [int(value) for value in attention_mask[index].detach().cpu().tolist()]
        # active counts [CLS], the word pieces, and [SEP], but not padding.
        active = sum(mask)
        if active > MAX_LENGTH:
            raise ValueError(
                f"Sentence {sentence['id']} is {active} WordPieces including special tokens; "
                f"the maximum is {MAX_LENGTH} and truncation is not allowed"
            )
        word_ids_batch.append(word_ids)
        masks.append(mask)
    # These are single sentences, not sentence pairs, so every segment id is 0.
    token_type_ids = torch.zeros_like(input_ids)
    # no_grad keeps the forward pass from building a graph for frozen BERT.
    with torch.no_grad():
        output = encoder(
            input_ids=input_ids.to(device),
            attention_mask=attention_mask.to(device),
            token_type_ids=token_type_ids.to(device),
            output_hidden_states=True,
        )
    hidden_states = _hidden_states(output, len(sentences), input_ids.shape[1])
    for index, sentence in enumerate(sentences):
        # bounds[word] is the half-open WordPiece range for that word.
        bounds = wordpiece_bounds(word_ids_batch[index], len(sentence["tokens"]), masks[index])
        sentence_states = [states[index] for states in hidden_states]
        for span in sentence["constituents"]:
            for layer, states in enumerate(sentence_states):
                layers[layer][cursor] = span_representation(states, bounds, span["start"], span["end"])
            label_id = label_ids[span["label"]]
            targets[cursor] = label_id
            rows.write(json.dumps({
                "end": span["end"],
                "label": span["label"],
                "label_id": label_id,
                "sentence_id": sentence["id"],
                "start": span["start"],
            }, sort_keys=True) + "\n")
            cursor += 1
    return cursor


def _as_long_tensor(value):
    """Accept either a torch tensor or the nested lists a test tokenizer returns."""
    if torch.is_tensor(value):
        return value.to(dtype=torch.long)
    return torch.tensor(value, dtype=torch.long)


def _hidden_states(output, batch_size, sequence_length):
    """Collect the 13 layer tensors from a BERT output or a small test double.

    Real BERT puts them on output.hidden_states when asked. A test encoder
    may return that tuple directly, or as the last element of its output.
    Index 0 is the embedding output. Indexes 1 through 12 are the blocks.
    """
    if getattr(output, "hidden_states", None) is not None:
        hidden = output.hidden_states
    elif isinstance(output, (tuple, list)) and len(output) == len(LAYERS) and torch.is_tensor(output[0]):
        hidden = output
    elif isinstance(output, (tuple, list)) and output and isinstance(output[-1], (tuple, list)):
        hidden = output[-1]
    else:
        raise ValueError("Encoder must return all 13 hidden states")
    if len(hidden) != len(LAYERS):
        raise ValueError("Encoder must return all 13 hidden states")
    for states in hidden:
        if not torch.is_tensor(states) or states.shape != (batch_size, sequence_length, HIDDEN_SIZE):
            raise ValueError("Each hidden state must have shape (batch, sequence, 768)")
    return hidden
