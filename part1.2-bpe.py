import argparse
import heapq
import sys
import time
from array import array
from pathlib import Path

import numpy as np

from data import iter_split_files


# Merges do not cross newlines. A token that contained a newline could not be
# used when each line is tokenized on its own, so 0x0a stays a single byte.
def load_corpus(input_dir):
    parts = []
    ends = []
    total = 0
    for path in iter_split_files(input_dir, "train"):
        for part in path.read_bytes().split(b"\n"):
            if not part:
                continue
            parts.append(part)
            total += len(part)
            ends.append(total)
    return b"".join(parts), ends


def _u32_array(values):
    buf = np.ascontiguousarray(values, dtype=np.uint32)
    out = array("I")
    out.frombytes(buf.tobytes())
    return out


def _i32_array(values):
    buf = np.ascontiguousarray(values, dtype=np.int32)
    out = array("i")
    out.frombytes(buf.tobytes())
    return out


def train_bpe(raw, ends, vocab_size):
    n = len(raw)
    if n == 0:
        raise ValueError("training corpus is empty")

    tokens = _u32_array(np.frombuffer(raw, dtype=np.uint8))
    nxt = _i32_array(np.arange(1, n + 1, dtype=np.int32))
    prv = _i32_array(np.arange(-1, n - 1, dtype=np.int32))
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
    index_started = time.time()

    for i in range(n):
        j = nxt[i]
        if j < 0:
            continue
        pair = (tokens[i], tokens[j])
        count = pair_counts.get(pair)
        if count is None:
            pair_counts[pair] = 1
            pair_head[pair] = i
        else:
            pair_counts[pair] = count + 1
            head = pair_head[pair]
            chain_next[i] = head
            chain_prev[head] = i
            pair_head[pair] = i

    heap = [(-count, a, b) for (a, b), count in pair_counts.items()]
    heapq.heapify(heap)
    dirty = set()

    vocab = [bytes((i,)) for i in range(256)]
    merges = []
    started = time.time()
    print(
        f"indexed pairs={len(pair_counts):,} in {started - index_started:.0f}s",
        file=sys.stderr,
    )

    def rebuild_heap():
        heap[:] = [(-c, x, y) for (x, y), c in pair_counts.items() if c > 0]
        heapq.heapify(heap)

    def pop_best():
        while heap:
            neg, x, y = heapq.heappop(heap)
            if pair_counts.get((x, y), 0) == -neg:
                return x, y, -neg
        return None

    def unlink(idx, pair):
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
        count = pair_counts.get(pair, 0) - 1
        if count > 0:
            pair_counts[pair] = count
        else:
            pair_counts.pop(pair, None)
        dirty.add(pair)

    def link(idx, pair):
        head = pair_head.get(pair, -1)
        chain_prev[idx] = -1
        chain_next[idx] = head
        if head >= 0:
            chain_prev[head] = idx
        pair_head[pair] = idx
        pair_counts[pair] = pair_counts.get(pair, 0) + 1
        dirty.add(pair)

    target = vocab_size - 256
    while len(vocab) < vocab_size:
        found = pop_best()
        if found is None:
            break
        a, b, count = found
        new_id = len(vocab)
        vocab.append(vocab[a] + vocab[b])
        merges.append((a, b))

        positions = []
        cursor = pair_head.get((a, b), -1)
        while cursor >= 0:
            positions.append(cursor)
            cursor = chain_next[cursor]
        positions.sort()
        pair_counts.pop((a, b), None)
        pair_head.pop((a, b), None)
        dirty.clear()

        for i in positions:
            if dead[i] or tokens[i] != a:
                continue
            j = nxt[i]
            if j < 0 or dead[j] or tokens[j] != b:
                continue

            left = prv[i]
            right = nxt[j]
            if left >= 0:
                left_pair = (tokens[left], a)
                if left_pair != (a, b):
                    unlink(left, left_pair)
            if right >= 0:
                right_pair = (b, tokens[right])
                if right_pair != (a, b):
                    unlink(j, right_pair)

            tokens[i] = new_id
            dead[j] = 1
            nxt[i] = right
            if right >= 0:
                prv[right] = i
            if left >= 0:
                link(left, (tokens[left], new_id))
            if right >= 0:
                link(i, (new_id, tokens[right]))

        for pair in dirty:
            changed = pair_counts.get(pair, 0)
            if changed > 0:
                heapq.heappush(heap, (-changed, pair[0], pair[1]))
        dirty.clear()

        done = len(merges)
        if done % 2000 == 0:
            rebuild_heap()
        if done % 1000 == 0 or done == target:
            print(
                f"merge {done}/{target} count={count} "
                f"elapsed={time.time() - started:.0f}s",
                file=sys.stderr,
            )

    return vocab, merges


def write_artifacts(output_dir, vocab, merges):
    output_dir.mkdir(parents=True, exist_ok=True)
    vocab_path = output_dir / "bpe_vocab.txt"
    merges_path = output_dir / "bpe_merges.txt"
    with vocab_path.open("w", encoding="utf-8", newline="\n") as handle:
        for token in vocab:
            handle.write(token.hex())
            handle.write("\n")
    with merges_path.open("w", encoding="utf-8", newline="\n") as handle:
        for left, right in merges:
            handle.write(vocab[left].hex())
            handle.write(" ")
            handle.write(vocab[right].hex())
            handle.write("\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--vocab-size", type=int, required=True)
    args = parser.parse_args()
    if args.vocab_size < 256:
        parser.error("--vocab-size must be at least 256")

    raw, ends = load_corpus(Path(args.input))
    print(
        f"corpus bytes={len(raw):,} lines={len(ends):,}",
        file=sys.stderr,
    )
    vocab, merges = train_bpe(raw, ends, args.vocab_size)
    write_artifacts(Path(args.output), vocab, merges)
    print(
        f"wrote {len(vocab)} vocab tokens and {len(merges)} merges",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
