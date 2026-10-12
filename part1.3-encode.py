import argparse
import heapq
from pathlib import Path


def load_merges(bpe_dir):
    """Load the vocabulary and the learned merge map."""
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
    """Apply the learned merges to one line of bytes."""
    # Lowest merge id is the earliest rule. Position breaks ties left to right,
    # which is how the trainer applied one rule.
    n = len(data)
    if n == 0:
        return []
    if n == 1:
        return [data[0]]

    ids = list(data)
    nxt = list(range(1, n))
    nxt.append(-1)
    prv = list(range(-1, n - 1))
    alive = bytearray(b"\x01") * n
    heap = []
    for i in range(n - 1):
        merged = merge_of.get((ids[i], ids[i + 1]))
        if merged is not None:
            heapq.heappush(heap, (merged, i))

    while heap:
        merged, i = heapq.heappop(heap)
        if not alive[i]:
            continue
        j = nxt[i]
        if j < 0 or not alive[j]:
            continue
        if merge_of.get((ids[i], ids[j])) != merged:
            continue
        ids[i] = merged
        alive[j] = 0
        k = nxt[j]
        nxt[i] = k
        if k >= 0:
            prv[k] = i
            nxt_merge = merge_of.get((merged, ids[k]))
            if nxt_merge is not None:
                heapq.heappush(heap, (nxt_merge, i))
        p = prv[i]
        if p >= 0 and alive[p]:
            prev_merge = merge_of.get((ids[p], merged))
            if prev_merge is not None:
                heapq.heappush(heap, (prev_merge, p))

    out = []
    i = 0
    while i >= 0:
        out.append(ids[i])
        i = nxt[i]
    return out


def encode(data, merge_of):
    """Encode bytes, keeping each newline as token 10."""
    # Newline is byte 10 and was never merged, so it stays its own token.
    out = []
    start = 0
    for i, byte in enumerate(data):
        if byte != 10:
            continue
        if i > start:
            out.extend(encode_line(data[start:i], merge_of))
        out.append(10)
        start = i + 1
    if start < len(data):
        out.extend(encode_line(data[start:], merge_of))
    return out


def main():
    """Encode one file into one token id per line."""
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
