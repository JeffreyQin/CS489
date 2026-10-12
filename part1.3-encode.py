"""
This file was written with assistance from generative AI. Uses include
    - Code for argument parsing and file path handling
    - Implementation of helper functions like load_merges and encode
    - Assistance on the linked-list implementation of encode_line
    - Code proofreading and improvements
"""

import argparse
import heapq
from pathlib import Path


def load_merges(bpe_dir):
    """Load vocabulary and learned merge mapping"""
    bpe_dir = Path(bpe_dir)
    vocab = [
        bytes.fromhex(line)
        for line in (bpe_dir / "bpe_vocab.txt").read_text(encoding="utf-8").splitlines()
    ]
    token_id = {token: index for index, token in enumerate(vocab)}
    merge_of = {}
    for line in (bpe_dir / "bpe_merges.txt").read_text(encoding="utf-8").splitlines():
        left_hex, right_hex = line.split(" ")
        left = bytes.fromhex(left_hex)
        right = bytes.fromhex(right_hex)
        merge_of[(token_id[left], token_id[right])] = token_id[left + right]
    return merge_of


def encode_line(data, merge_of):
    """
    Encode a line of bytes by applying learned merges
    Prioritize merges by lowest merge id, then left to right position to break ties
    """

    n = len(data)
    if n == 0:
        return []
    if n == 1:
        return [data[0]]

    # each live token points at its neighbors (-1 for placeholder)
    ids = list(data)
    next = list(range(1, n))
    next.append(-1)
    prev = list(range(-1, n - 1))

    # 1 means the token is still in the line (alive). 0 means it was removed (dead).
    alive = bytearray(b"\x01") * n

    # queue every pair that has a learned merge, as (merged id, position).
    heap = []
    for i in range(n - 1):
        merged = merge_of.get((ids[i], ids[i + 1]))
        if merged is not None:
            heapq.heappush(heap, (merged, i))

    while heap:
        merged, i = heapq.heappop(heap)

        j = next[i]
        # queued pair is no longer neighboring live tokens.
        if not alive[i] or j < 0 or merge_of.get((ids[i], ids[j])) != merged:
            continue

        # left token stays alive and holds the merged id, right token is marked as dead
        ids[i] = merged
        alive[j] = 0
        k = next[j]
        next[i] = k
        if k >= 0:
            prev[k] = i
            next_merge = merge_of.get((merged, ids[k]))
            if next_merge is not None:
                heapq.heappush(heap, (next_merge, i))

        # left neighbor forms new pair with merged token.
        p = prev[i]
        if p >= 0 and alive[p]:
            prev_merge = merge_of.get((ids[p], merged))
            if prev_merge is not None:
                heapq.heappush(heap, (prev_merge, p))

    # read tokens that are still linked together
    out = []
    i = 0
    while i >= 0:
        out.append(ids[i])
        i = next[i]
    return out


def encode(data, merge_of):
    """Encode corpus by encoding each line separately and appending the newline token (byte 10)"""

    encoded = []
    start = 0
    for i, byte in enumerate(data):
        if byte != 10:
            continue
        if i > start:
            encoded.extend(encode_line(data[start:i], merge_of))
        encoded.append(10)
        start = i + 1
    if start < len(data):
        encoded.extend(encode_line(data[start:], merge_of))
    return encoded


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--bpe", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    merge_of = load_merges(args.bpe)
    data = Path(args.input).read_bytes()
    ids = encode(data, merge_of)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for token_id in ids:
            handle.write(str(token_id))
            handle.write("\n")


if __name__ == "__main__":
    main()
