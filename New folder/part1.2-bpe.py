"""
Current implementation of BPE uses a linked list to store bytes from the corpus for efficiency.
Merges only change the two neighbor pointers around the removed tokens, so the location of other tokens are untouched.

In current implementation, merges do not cross newline boundaries, please see write-up for rationale.

This file was written with assistance from generative AI. 
All AI-generated code were read-over and improved by hand for correctness.

Uses of AI include
    - Implementation of the linked list optimization on top of the original BPE algorithm
    - Implementation of helper functions like load_corpus, _u32_array, and _i32_array
    - Starter code draft and assistance with proofreading for the core algorithm
"""


import argparse
import heapq
import sys
from array import array
from pathlib import Path
import numpy as np

from data import iter_split_files

def load_corpus(input_dir):
    """Read and concatenated content from all languages' train.txt"""
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
    """Copy values into an unsigned 32-bit array."""
    buf = np.ascontiguousarray(values, dtype=np.uint32)
    out = array("I")
    out.frombytes(buf.tobytes())
    return out


def _i32_array(values):
    """Copy values into a signed 32-bit array."""
    buf = np.ascontiguousarray(values, dtype=np.int32)
    out = array("i")
    out.frombytes(buf.tobytes())
    return out


def train_bpe(raw, ends, vocab_size):
    """
    Main training code for BPE
    merge the most frequent pair until the vocabulary reaches vocab_size
    """
    n = len(raw)

    # each live token points at its neighbors. -1 means there is no neighbor.
    tokens = _u32_array(np.frombuffer(raw, dtype=np.uint8))
    next = _i32_array(np.arange(1, n + 1, dtype=np.int32))
    prev = _i32_array(np.arange(-1, n - 1, dtype=np.int32))
    next[n - 1] = -1

    # cut the list at every line so a merge cannot cross a newline.
    for end in ends:
        next[end - 1] = -1
        if end < n:
            prev[end] = -1

    # marks removed positions. The chains list every position of one pair.
    dead = bytearray(n)
    chain_next = array("i", [-1]) * n
    chain_prev = array("i", [-1]) * n
    pair_counts = {}
    pair_head = {}

    # count every adjacent pair and link its positions together.
    for i in range(n):
        j = next[i]
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

    # main heap for finding most frequent pair
    # ranked by count, followed by pair id
    heap = [(-count, a, b) for (a, b), count in pair_counts.items()]
    heapq.heapify(heap)

    # pairs whose counts changed in the current merge
    dirty = set()
    
    # vocab (initalized to base vocab)
    vocab = [bytes((i,)) for i in range(256)]
    
    # learned pairs in the order they were merged
    merges = []

    def rebuild_heap():
        """Rebuild the pair heap from the current counts."""
        heap[:] = [(-c, x, y) for (x, y), c in pair_counts.items() if c > 0]
        heapq.heapify(heap)

    def pop_best():
        """Return the most frequent pair"""
        while heap:
            neg, x, y = heapq.heappop(heap)
            # skip heap entry is count is no longer current
            if pair_counts.get((x, y), 0) == -neg:
                return x, y
        return None

    def unlink(idx, pair):
        """Remove one occurrence of a pair from its chain."""
        # splice this position out of the pair's chain.
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
        # decrease the count and remember that this pair needs a new heap entry.
        count = pair_counts.get(pair, 0) - 1
        if count > 0:
            pair_counts[pair] = count
        else:
            pair_counts.pop(pair, None)
        dirty.add(pair)

    def link(idx, pair):
        """Record one new occurrence of a pair."""
        # insert this position at the front of the pair's chain.
        head = pair_head.get(pair, -1)
        chain_prev[idx] = -1
        chain_next[idx] = head
        if head >= 0:
            chain_prev[head] = idx
        pair_head[pair] = idx
        # increase the count and remember that this pair needs a new heap entry.
        pair_counts[pair] = pair_counts.get(pair, 0) + 1
        dirty.add(pair)

    # bpe merge loop
    while len(vocab) < vocab_size:
        found = pop_best()

        # no more pairs to merge
        if found is None:
            break

        # add merged token to vocab
        a, b = found
        new_id = len(vocab)
        vocab.append(vocab[a] + vocab[b])
        merges.append((a, b))

        # gather every occurrence of the pair left to right, then forget the pair.
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
            # skip a copy that an earlier overlap in this merge already used.
            if dead[i] or tokens[i] != a:
                continue
            j = next[i]
            if j < 0 or dead[j] or tokens[j] != b:
                continue

            # remove the neighboring pairs that this merge destroys.
            left = prev[i]
            right = next[j]
            if left >= 0:
                left_pair = (tokens[left], a)
                if left_pair != (a, b):
                    unlink(left, left_pair)
            if right >= 0:
                right_pair = (b, tokens[right])
                if right_pair != (a, b):
                    unlink(j, right_pair)

            # keep the new token at i and link it to the tokens on either side.
            tokens[i] = new_id
            dead[j] = 1
            next[i] = right
            if right >= 0:
                prev[right] = i
            if left >= 0:
                link(left, (tokens[left], new_id))
            if right >= 0:
                link(i, (new_id, tokens[right]))

        # put the pairs whose counts changed back on the heap.
        for pair in dirty:
            changed = pair_counts.get(pair, 0)
            if changed > 0:
                heapq.heappush(heap, (-changed, pair[0], pair[1]))
        dirty.clear()

        # drop stale heap entries every 2000 merges.
        if len(merges) % 2000 == 0:
            rebuild_heap()

    return vocab, merges


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--vocab-size", type=int, required=True)
    args = parser.parse_args()

    raw, ends = load_corpus(Path(args.input))
    vocab, merges = train_bpe(raw, ends, args.vocab_size)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    vocab_path = output_dir / "bpe_vocab.txt"
    merges_path = output_dir / "bpe_merges.txt"

    # write to vocab file
    with vocab_path.open("w", encoding="utf-8", newline="\n") as handle:
        for token in vocab:
            handle.write(token.hex())
            handle.write("\n")

    # write to merged tokens file
    with merges_path.open("w", encoding="utf-8", newline="\n") as handle:
        for left, right in merges:
            handle.write(vocab[left].hex())
            handle.write(" ")
            handle.write(vocab[right].hex())
            handle.write("\n")

    print(
        f"wrote {len(vocab)} vocab tokens and {len(merges)} merges",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
