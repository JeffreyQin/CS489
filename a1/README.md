# Set-up

Run the following to install the environment and download the corpus. The data is saved to `data/`.

```bash
uv sync
uvx --from huggingface_hub hf download compling/cs489-f26-a1 --repo-type dataset --local-dir data
```

# Part 1: Data Cleaning and byte-BPE Tokenizer

## Subtask 1

Run the following to clean the raw Wikipedia dumps and print raw and cleaned byte counts. The cleaner reads `data/raw/`. Cleaned text is saved to `artifacts/part1/cleaned/`. The byte counts are printed to the console.

```bash
uv run python part1.1-clean.py --input data/raw/ --output artifacts/part1/cleaned/
uv run python part1.1-stats.py --raw-input data/raw/ --clean-input artifacts/part1/cleaned/
```

## Subtask 2

Run the following to train byte-level BPE at vocabulary size 50,000 on the cleaned training text. It reads `artifacts/part1/cleaned/`, and saves trained vocab `bpe_vocab.txt` and `bpe_merges.txt` to `artifacts/part1/bpe/`.

```bash
uv run python part1.2-bpe.py --input artifacts/part1/cleaned --output artifacts/part1/bpe/ --vocab-size 50000
```

## Subtask 3

Run the following to encode one cleaned file with the trained BPE model and decode the token ids back to text. The encoder reads `artifacts/part1/cleaned/fi/dev.txt` and `artifacts/part1/bpe/`. Token ids are saved to `artifacts/part1/encoded/`, and the restored text to `artifacts/part1/decoded/`.

```bash
uv run python part1.3-encode.py --input artifacts/part1/cleaned/fi/dev.txt \
    --bpe artifacts/part1/bpe/ --output artifacts/part1/encoded/fi/dev.txt
uv run python part1.3-decode.py --input artifacts/part1/encoded/fi/dev.txt \
    --bpe artifacts/part1/bpe/ --output artifacts/part1/decoded/fi/dev.txt
```

## Subtask 4

Run the following to train a unigram tokenizer at vocabulary size 50,000 on the cleaned training text. It reads `artifacts/part1/cleaned/`. Model is saved to `artifacts/part1/unigram/unigram.model`.

```bash
uv run python part1.4-unigram.py --input artifacts/part1/cleaned \
    --output artifacts/part1/unigram/ --vocab-size 50000
```

## Subtask 5

Run the following to train BPE and unigram at vocabulary sizes 5,000, 10,000, 20,000, and 50,000 and plot tokens per byte. It reads the cleaned training and dev text in `artifacts/part1/cleaned/`. The plot and the sweep tokenizers are saved to `artifacts/part1/analysis/`.

```bash
uv run python part1.5-analysis.py --input artifacts/part1/cleaned \
    --output artifacts/part1/analysis/
```

# Part 2: Optimize Bits-per-Character

## Subtask 1

Run the following to train the baseline Transformer on the cleaned text using the 50,000-token BPE vocabulary, then print dev bits per character. It reads `artifacts/part1/cleaned/` and the BPE files in `artifacts/part1/bpe/`. The checkpoint, config, and tokenizer are saved to `artifacts/part2/`.

```bash
uv run python part2.py --input artifacts/part1/cleaned \
    --output artifacts/part2/ --vocab-size 50000
uv run python part2_eval_bpc.py --config artifacts/part2/config.json \
    --model artifacts/part2/model.pt --ids artifacts/part2/dev.bin \
    --text artifacts/part2/dev.txt
```

## Subtask 2

Run the following to train the submitted Transformer on the cleaned text using the 50,000-token unigram model, then print dev bits per character. This replaces the baseline files in `artifacts/part2/`. It reads `artifacts/part1/cleaned/` and `artifacts/part1/unigram/unigram.model`. The checkpoint, config, and tokenizer are saved to `artifacts/part2/`.

```bash
uv run python part2.py --input artifacts/part1/cleaned \
    --output artifacts/part2/ --vocab-size 50000 \
    --algorithm unigram --tokenizer artifacts/part1/unigram
uv run python part2_eval_bpc.py --config artifacts/part2/config.json \
    --model artifacts/part2/model.pt --ids artifacts/part2/dev.bin \
    --text artifacts/part2/dev.txt
```

## Subtask 3

Run the following to train the Transformer for BPE and unigram at four vocabulary sizes and plot dev bits per character. The sweeps read the cleaned text and the saved tokenizers in `artifacts/part1/analysis/`. Each checkpoint is saved to `artifacts/part2/sweep/`, and the chart to `artifacts/part2/analysis/`.

```bash
uv run python part2_analysis/sweep.py --input artifacts/part1/cleaned \
    --output artifacts/part2/sweep/ --vocab-sizes 5000,10000,20000,50000 --algorithm bpe
uv run python part2_analysis/sweep.py --input artifacts/part1/cleaned \
    --output artifacts/part2/sweep/ --vocab-sizes 5000,10000,20000,50000 --algorithm unigram
uv run python part2_analysis/plot_bpc.py --runs artifacts/part2/sweep/ \
    --output artifacts/part2/analysis/
```

# Part 3: N-gram Language Model

## Subtask 1

Run the following to train an unsmoothed n-gram model on the cleaned training text and print dev perplexity. Repeat for `--order` 2, 3, 4, and 5. It reads `artifacts/part1/cleaned/` and the BPE files in `artifacts/part1/bpe/`.

```bash
uv run python part3_ngram_lm.py --input artifacts/part1/cleaned \
    --bpe artifacts/part1/bpe/ --order 2 --smoothing none
```

## Subtask 2

Run the following to train an add-α n-gram model on the cleaned training text and print dev perplexity. Change `--order` and `--alpha` for each pair. It reads `artifacts/part1/cleaned/` and the BPE files in `artifacts/part1/bpe/`.

```bash
uv run python part3_ngram_lm.py --input artifacts/part1/cleaned \
    --bpe artifacts/part1/bpe/ --order 2 --smoothing add-alpha --alpha 0.01
```

## Subtask 3

Run the following to train the submitted n-gram model on the cleaned training text and print its dev perplexity and bits per character. By default, `part3.py` uses `ORDER = 2` and `ALPHA = 0.01`. It reads `artifacts/part1/cleaned/` and the BPE files in `artifacts/part1/bpe/`. The trained model `model.json.gz` is saved to `artifacts/part3/`.

```bash
uv run python part3.py --input artifacts/part1/cleaned \
    --bpe artifacts/part1/bpe/ --output artifacts/part3/
```
