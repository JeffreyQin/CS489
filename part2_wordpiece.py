"""Byte-level WordPiece.

Words are split on spaces and tabs. Multibyte characters that occur at least
50 times are tokens before any merge, most frequent first, and every byte stays
available as a fallback. Each merge then maximizes
count(pair) / (count(left) * count(right)), weighted by word frequency. A pair
must occur at least 50 times. Merges stay inside one
word and do not cross newlines. Encoding is longest match, so
decode(encode(text)) reproduces the text.
"""

import argparse
import heapq
import importlib.util
import sys
import time
from array import array
from pathlib import Path


MIN_COUNT = 50
CHAR_MIN_FREQ = 50
PREFIX_SIZES = (5000, 10000, 20000, 50000)


def _bpe():
    """Load the Part 1.2 helpers."""
    path = Path(__file__).resolve().parent / "part1.2-bpe.py"
    spec = importlib.util.spec_from_file_location("part1_2_bpe", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _word_counts(raw, ends):
    """Count whitespace-delimited words. Spaces and tabs stay out of the words."""
    from collections import Counter

    counts = Counter()
    start = 0
    for end in ends:
        line = raw[start:end]
        index = 0
        size = len(line)
        while index < size:
            if line[index] in (32, 9):
                index += 1
                continue
            cursor = index + 1
            while cursor < size and line[cursor] not in (32, 9):
                cursor += 1
            counts[line[index:cursor]] += 1
            index = cursor
        start = end
    return counts


def _symbol_corpus(word_counts):
    """Map each word to characters, with bytes left for rare single-byte symbols."""
    import numpy as np
    from collections import Counter

    char_counts = Counter()
    for word, freq in word_counts.items():
        for char in word.decode("utf-8"):
            char_counts[char] += freq
    characters = [
        char
        for char in char_counts
        if len(char.encode("utf-8")) > 1 and char_counts[char] >= CHAR_MIN_FREQ
    ]
    characters.sort(key=lambda char: (-char_counts[char], char.encode("utf-8")))
    vocab = [bytes((index,)) for index in range(256)]
    char_id = {}
    for char in characters:
        char_id[char] = len(vocab)
        vocab.append(char.encode("utf-8"))

    symbols = []
    weights = []
    word_ends = []
    for word, freq in word_counts.items():
        for char in word.decode("utf-8"):
            token_id = char_id.get(char)
            if token_id is None:
                encoded = char.encode("utf-8")
                symbols.extend(encoded)
                weights.extend([freq] * len(encoded))
            else:
                symbols.append(token_id)
                weights.append(freq)
        word_ends.append(len(symbols))
    return (
        vocab,
        np.asarray(symbols, dtype=np.uint32),
        word_ends,
        np.asarray(weights, dtype=np.int64),
    )


def train_wordpiece(raw, ends, vocab_size, min_count=MIN_COUNT):
    """Merge highest-scoring pairs until the vocabulary reaches vocab_size."""
    import numpy as np

    if len(raw) == 0:
        raise ValueError("training corpus is empty")
    if vocab_size < 256:
        raise ValueError("vocab_size must be at least 256")
    word_counts = _word_counts(raw, ends)
    vocab, symbols, ends, weights = _symbol_corpus(word_counts)
    print(
        f"unique words={len(word_counts):,} symbols={len(symbols):,} "
        f"characters={len(vocab) - 256:,}",
        file=sys.stderr,
    )
    if len(vocab) >= vocab_size:
        return vocab[:vocab_size], []

    bpe = _bpe()
    n = len(symbols)
    tokens = bpe._u32_array(symbols)
    nxt = bpe._i32_array(np.arange(1, n + 1, dtype=np.int32))
    prv = bpe._i32_array(np.arange(-1, n - 1, dtype=np.int32))
    nxt[n - 1] = -1
    for end in ends:
        nxt[end - 1] = -1
        if end < n:
            prv[end] = -1

    dead = bytearray(n)
    chain_next = array("i", [-1]) * n
    chain_prev = array("i", [-1]) * n
    pair_counts = {}
    pair_head = {}
    pairs_with = [set() for _ in range(vocab_size)]
    token_counts = [0] * vocab_size
    weighted = np.bincount(symbols, weights=weights, minlength=len(vocab))
    for index, value in enumerate(weighted):
        token_counts[index] = int(value)

    index_started = time.time()
    for i in range(n):
        j = nxt[i]
        if j < 0:
            continue
        pair = (int(tokens[i]), int(tokens[j]))
        weight = int(weights[i])
        count = pair_counts.get(pair)
        if count is None:
            pair_counts[pair] = weight
            pair_head[pair] = i
            pairs_with[pair[0]].add(pair)
            pairs_with[pair[1]].add(pair)
        else:
            pair_counts[pair] = count + weight
            head = pair_head[pair]
            chain_next[i] = head
            chain_prev[head] = i
            pair_head[pair] = i

    heap = []
    dirty = set()
    merges = []
    started = time.time()
    print(
        f"indexed pairs={len(pair_counts):,} in {started - index_started:.0f}s",
        file=sys.stderr,
    )

    def push_pair(pair):
        """Push the current score of one pair."""
        count = pair_counts.get(pair, 0)
        if count < min_count:
            return
        left, right = pair
        left_count = token_counts[left]
        right_count = token_counts[right]
        if left_count <= 0 or right_count <= 0:
            return
        heapq.heappush(
            heap,
            (-(count / (left_count * right_count)), left, right, count, left_count, right_count),
        )

    def rebuild_heap():
        """Drop stale heap entries and score every live pair again."""
        fresh = []
        for (left, right), count in pair_counts.items():
            if count < min_count:
                continue
            left_count = token_counts[left]
            right_count = token_counts[right]
            if left_count <= 0 or right_count <= 0:
                continue
            fresh.append(
                (-(count / (left_count * right_count)), left, right, count, left_count, right_count)
            )
        heap[:] = fresh
        heapq.heapify(heap)

    def pop_best():
        """Return the highest-scoring pair whose counts are still current."""
        while heap:
            _neg, left, right, count, left_count, right_count = heapq.heappop(heap)
            current = pair_counts.get((left, right), 0)
            if current < min_count:
                continue
            if (
                current == count
                and token_counts[left] == left_count
                and token_counts[right] == right_count
            ):
                return left, right, current
        return None

    def unlink(idx, pair):
        """Remove one occurrence of a pair from its chain."""
        left = chain_prev[idx]
        right = chain_next[idx]
        if left >= 0:
            chain_next[left] = right
        elif right >= 0:
            pair_head[pair] = right
        else:
            pair_head.pop(pair, None)
        if right >= 0:
            chain_prev[right] = left
        chain_prev[idx] = -1
        chain_next[idx] = -1
        count = pair_counts.get(pair, 0) - int(weights[idx])
        if count > 0:
            pair_counts[pair] = count
        else:
            pair_counts.pop(pair, None)
            pairs_with[pair[0]].discard(pair)
            pairs_with[pair[1]].discard(pair)
        dirty.add(pair)

    def link(idx, pair):
        """Record one new occurrence of a pair."""
        head = pair_head.get(pair, -1)
        chain_prev[idx] = -1
        chain_next[idx] = head
        if head >= 0:
            chain_prev[head] = idx
        pair_head[pair] = idx
        pair_counts[pair] = pair_counts.get(pair, 0) + int(weights[idx])
        pairs_with[pair[0]].add(pair)
        pairs_with[pair[1]].add(pair)
        dirty.add(pair)

    rebuild_heap()
    target = vocab_size - len(vocab)
    while len(vocab) < vocab_size:
        found = pop_best()
        if found is None:
            break
        left_id, right_id, _count = found
        new_id = len(vocab)
        positions = []
        cursor = pair_head.get((left_id, right_id), -1)
        while cursor >= 0:
            positions.append(cursor)
            cursor = chain_next[cursor]
        positions.sort()
        pair_counts.pop((left_id, right_id), None)
        pair_head.pop((left_id, right_id), None)
        pairs_with[left_id].discard((left_id, right_id))
        pairs_with[right_id].discard((left_id, right_id))
        dirty.clear()

        applied = 0
        for i in positions:
            if dead[i] or tokens[i] != left_id:
                continue
            j = nxt[i]
            if j < 0 or dead[j] or tokens[j] != right_id:
                continue
            left = prv[i]
            right = nxt[j]
            if left >= 0:
                left_pair = (int(tokens[left]), left_id)
                if left_pair != (left_id, right_id):
                    unlink(left, left_pair)
            if right >= 0:
                right_pair = (right_id, int(tokens[right]))
                if right_pair != (left_id, right_id):
                    unlink(j, right_pair)
            tokens[i] = new_id
            dead[j] = 1
            nxt[i] = right
            if right >= 0:
                prv[right] = i
            if left >= 0:
                link(left, (int(tokens[left]), new_id))
            if right >= 0:
                link(i, (new_id, int(tokens[right])))
            applied += int(weights[i])

        if applied == 0:
            for pair in dirty:
                push_pair(pair)
            dirty.clear()
            continue

        vocab.append(vocab[left_id] + vocab[right_id])
        merges.append((left_id, right_id))
        token_counts[left_id] -= applied
        token_counts[right_id] -= applied
        token_counts[new_id] = applied
        dirty.update(pairs_with[left_id])
        dirty.update(pairs_with[right_id])
        done = len(merges)
        if done % 2000 == 0:
            rebuild_heap()
        else:
            for pair in dirty:
                push_pair(pair)
        dirty.clear()
        if done % 1000 == 0 or done == target:
            print(
                f"merge {done}/{target} elapsed={time.time() - started:.0f}s",
                file=sys.stderr,
            )

    if len(vocab) != vocab_size:
        raise RuntimeError(f"WordPiece produced {len(vocab)} tokens, expected {vocab_size}")
    return vocab, merges


def write_artifacts(output_dir, vocab, merges):
    """Write wordpiece_vocab.txt and wordpiece_merges.txt."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "wordpiece_vocab.txt").open("w", encoding="utf-8", newline="\n") as handle:
        for token in vocab:
            handle.write(token.hex())
            handle.write("\n")
    with (output_dir / "wordpiece_merges.txt").open("w", encoding="utf-8", newline="\n") as handle:
        for left, right in merges:
            handle.write(vocab[left].hex())
            handle.write(" ")
            handle.write(vocab[right].hex())
            handle.write("\n")


def write_prefixes(output_dir, vocab, merges):
    """Write each saved vocabulary size as a prefix of this training run."""
    output_dir = Path(output_dir)
    base = len(vocab) - len(merges)
    for size in PREFIX_SIZES:
        if size > len(vocab):
            continue
        keep = max(0, size - base)
        write_artifacts(output_dir / f"v{size}", vocab[:size], merges[:keep])


def main():
    """Train byte-level WordPiece and write the vocabulary prefixes."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--vocab-size", type=int, required=True)
    args = parser.parse_args()
    if args.vocab_size < 256:
        parser.error("--vocab-size must be at least 256")

    bpe = _bpe()
    raw, ends = bpe.load_corpus(Path(args.input))
    print(f"corpus bytes={len(raw):,} lines={len(ends):,}", file=sys.stderr)
    vocab, merges = train_wordpiece(raw, ends, args.vocab_size)
    write_prefixes(Path(args.output), vocab, merges)
    print(f"wrote {len(vocab)} vocab tokens and {len(merges)} merges", file=sys.stderr)


if __name__ == "__main__":
    main()
