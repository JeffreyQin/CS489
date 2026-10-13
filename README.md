# CS 489 Assignment 1

Commands to reproduce the cleaning, tokenizers, Transformer runs, and n-gram model. Run every command from the project root with `uv run python`. Python 3.11 or later is required. Only Part 2 needs a GPU.

`part2_transformer_lm.py`, `part2_train.py`, `part2_eval_bpc.py`, `part3_artifacts.py`, and `data.py` are fixed. Do not modify them. The grader restores those files before re-running an entry point.

## Setup

```bash
uv sync
uvx --from huggingface_hub hf download compling/cs489-f26-a1 \
    --repo-type dataset --local-dir data
```

`uv sync` installs the locked environment from `pyproject.toml` and `uv.lock`. The download creates `data/raw/` and `data/clean_reference/`.

Languages are read in sorted relative-path order: `ar`, `de`, `fi`, `hu`, `ru`, `vi`, `zh`.

## Part 1

### 1.1 Clean the dumps

```bash
uv run python part1.1-clean.py --input data/raw/ --output artifacts/part1/cleaned/
uv run python part1.1-stats.py --raw-input data/raw/ --clean-input artifacts/part1/cleaned/
```

The cleaner writes each file to the same relative path under the output directory. `data/raw/fi/train.txt` becomes `artifacts/part1/cleaned/fi/train.txt`. The stats script prints raw and cleaned train and dev byte counts and does not write a file. Later parts read `artifacts/part1/cleaned/`. Do not submit that directory.

### 1.2 Train byte-level BPE

```bash
uv run python part1.2-bpe.py --input artifacts/part1/cleaned \
    --output artifacts/part1/bpe/ --vocab-size 50000
```

This concatenates every cleaned `train.txt` and trains to 50,000 tokens. Merges stay inside a line. It writes:

- `artifacts/part1/bpe/bpe_vocab.txt`, one lowercase hex token per line, in id order
- `artifacts/part1/bpe/bpe_merges.txt`, one merge per line, in learning order

Submit both files. This training run is long.

### 1.3 Encode and decode

```bash
uv run python part1.3-encode.py --input artifacts/part1/cleaned/fi/dev.txt \
    --bpe artifacts/part1/bpe/ --output artifacts/part1/encoded/fi/dev.txt
uv run python part1.3-decode.py --input artifacts/part1/encoded/fi/dev.txt \
    --bpe artifacts/part1/bpe/ --output artifacts/part1/decoded/fi/dev.txt
diff artifacts/part1/cleaned/fi/dev.txt artifacts/part1/decoded/fi/dev.txt
```

The encoder writes one decimal token id per line. Token 10 is the newline. The decoder writes the original bytes. `diff` should print nothing. Replace `fi/dev.txt` with any other cleaned file to check it. These encoded and decoded files are not part of the submission.

### 1.4 Train the unigram tokenizer

```bash
uv run python part1.4-unigram.py --input artifacts/part1/cleaned \
    --output artifacts/part1/unigram/ --vocab-size 50000
```

SentencePiece writes `artifacts/part1/unigram/unigram.model`. It also writes `unigram.vocab` beside it. Submit `unigram.model`. The model is lossless: decoding the token ids returns the original text, including line breaks.

### 1.5 Compression sweep

```bash
uv run python part1.5-analysis.py --input artifacts/part1/cleaned \
    --output artifacts/part1/analysis/
```

This retrains BPE once to 50,000 and saves each smaller vocabulary as a prefix, then trains a separate unigram model at 5,000, 10,000, 20,000, and 50,000. It does not reuse the files from sections 1.2 and 1.4. It writes:

- `artifacts/part1/analysis/compression.png`
- `artifacts/part1/analysis/compression_ratios.tsv`
- `artifacts/part1/analysis/bpe/v{5000,10000,20000,50000}/`
- `artifacts/part1/analysis/unigram/v{5000,10000,20000,50000}/unigram.model`

Part 2 loads the smaller vocabularies from those directories. The plot is for the writeup. The analysis directory itself is not in the submission tree.

## Part 2

`part2.py` does not train a tokenizer. It copies a saved vocabulary, encodes the cleaned train and dev text, trains the fixed Transformer, and writes the checkpoint. One run takes about 15 to 30 minutes on a Colab T4. The seed is 489. The last digits of dev bits per character can still differ across GPU runs.

The submitted directory is `artifacts/part2/`. Run the baseline first if you want its dev number, then run the Unigram command below into that same directory so the submitted checkpoint is the better one.

### Baseline: byte BPE, vocabulary 50,000

```bash
uv run python part2.py --input artifacts/part1/cleaned \
    --output artifacts/part2/ --vocab-size 50000
uv run python part2_eval_bpc.py --config artifacts/part2/config.json \
    --model artifacts/part2/model.pt --ids artifacts/part2/dev.bin \
    --text artifacts/part2/dev.txt
```

With no `--algorithm`, this loads `artifacts/part1/bpe/`. To keep this checkpoint, use a different `--output` before the Unigram run.

### Submitted run: unigram, vocabulary 50,000

```bash
uv run python part2.py --input artifacts/part1/cleaned \
    --output artifacts/part2/ --vocab-size 50000 \
    --algorithm unigram --tokenizer artifacts/part1/unigram
uv run python part2_eval_bpc.py --config artifacts/part2/config.json \
    --model artifacts/part2/model.pt --ids artifacts/part2/dev.bin \
    --text artifacts/part2/dev.txt
```

`--tokenizer` forces `artifacts/part1/unigram/`. Without it, an existing `artifacts/part1/analysis/unigram/v50000/` directory would be used instead.

Either `part2.py` command writes these files under its `--output` directory:

- `tokenizer/`, a copy of the tokenizer needed to encode new text
- `dev.txt`, the concatenated cleaned dev text
- `train.bin` and `dev.bin`, uint32 token ids
- `model.pt`, the Transformer checkpoint
- `run_config.json`, the fixed model and training settings
- `config.json`, `run_config.json` plus the tokenizer section
- `dev_metrics.json`, the trainer's final dev metrics

The zip needs `model.pt`, `config.json`, and `tokenizer/` from the Unigram run. Report the bits per character printed by `part2_eval_bpc.py`.

### Bits per character against vocabulary size

Train WordPiece once before its sweep. This writes `v5000`, `v10000`, `v20000`, and `v50000` under the output directory. `part2_wordpiece.py` is a local trainer and is not in the submission tree.

```bash
uv run python part2_wordpiece.py --input artifacts/part1/cleaned \
    --output artifacts/part1/analysis/wordpiece --vocab-size 50000
```

Each sweep command trains four Transformer runs for one algorithm. Run Part 1.5 first so the BPE and unigram sizes exist. Each finished size is skipped when its `dev_metrics.json` is already present.

```bash
uv run python part2_analysis/sweep.py --input artifacts/part1/cleaned \
    --output artifacts/part2/sweep/ --vocab-sizes 5000,10000,20000,50000 --algorithm bpe
uv run python part2_analysis/sweep.py --input artifacts/part1/cleaned \
    --output artifacts/part2/sweep/ --vocab-sizes 5000,10000,20000,50000 --algorithm unigram
uv run python part2_analysis/sweep.py --input artifacts/part1/cleaned \
    --output artifacts/part2/sweep/ --vocab-sizes 5000,10000,20000,50000 --algorithm wordpiece
```

A sweep stores one run at `artifacts/part2/sweep/<algorithm>/v<size>/` and writes a one-algorithm chart in `artifacts/part2/sweep/`. After all three algorithms have finished, this draws the combined chart from those metrics:

```bash
uv run python part2_analysis/plot_bpc.py --runs artifacts/part2/sweep/ \
    --output artifacts/part2/analysis/
```

That writes `artifacts/part2/analysis/bpc_vs_v.png` and `bpc_vs_v.tsv`. Omitting `--runs` redraws the chart from the dev numbers already stored in `part2_analysis/plot_bpc.py` and does not train anything.

`part2.py` looks up vocabularies as follows:

- byte BPE at 50,000: `artifacts/part1/bpe/`
- unigram at 50,000, when `--tokenizer` is omitted and the analysis directory is absent: `artifacts/part1/unigram/`
- every other size and algorithm: `artifacts/part1/analysis/<algorithm>/v<size>/`

## Part 3

Both scoring scripts use the submitted 50,000-token BPE model. Each cleaned line is encoded on its own. N-grams do not cross lines. Token 50,000 is the end-of-line token and is scored. No beginning-of-line token is used, so the vocabulary size is 50,001.

### 3.1 Unsmoothed n-gram

```bash
uv run python part3_ngram_lm.py --input artifacts/part1/cleaned \
    --bpe artifacts/part1/bpe/ --order 2 --smoothing none
```

Replace `2` with `3`, `4`, or `5`. The script trains on the cleaned train files and prints `dev perplexity=` on standard output. An unseen dev n-gram prints `dev perplexity=inf` and names the missing n-gram on standard error. This command does not write a file.

### 3.2 Add-alpha

```bash
uv run python part3_ngram_lm.py --input artifacts/part1/cleaned \
    --bpe artifacts/part1/bpe/ --order 2 --smoothing add-alpha --alpha 0.01
```

Change `--order` and `--alpha` for each pair you want to score. Each run counts the training n-grams again. Alpha is applied only at scoring time. This command does not write a file.

### 3.3 Submitted model

`part3.py` trains the pair named by `ORDER` and `ALPHA` at the top of that file. They are currently `ORDER = 2` and `ALPHA = 0.004`. Change those two lines to submit a different pair, then run:

```bash
uv run python part3.py --input artifacts/part1/cleaned \
    --bpe artifacts/part1/bpe/ --output artifacts/part3/
```

This writes `artifacts/part3/model.json.gz` through `part3_artifacts.save_ngram_model`. The file records the order, alpha, vocabulary size, and count tables.

### 3.4 Per-language perplexity

Save one count file per order. Alpha is stored in the file and does not change the counts, so leave `ALPHA` alone and set `ORDER` to 2, then 3, then 4, then 5 before each command:

```bash
uv run python part3.py --input artifacts/part1/cleaned \
    --bpe artifacts/part1/bpe/ --output artifacts/part3/counts/order2/
uv run python part3.py --input artifacts/part1/cleaned \
    --bpe artifacts/part1/bpe/ --output artifacts/part3/counts/order3/
uv run python part3.py --input artifacts/part1/cleaned \
    --bpe artifacts/part1/bpe/ --output artifacts/part3/counts/order4/
uv run python part3.py --input artifacts/part1/cleaned \
    --bpe artifacts/part1/bpe/ --output artifacts/part3/counts/order5/
```

Set `ORDER` and `ALPHA` back to the submitted pair before running the section 3.3 command. Then score Finnish and Mandarin at each saved order:

```bash
uv run python part3_analysis/eval_languages.py \
    --input artifacts/part1/cleaned --bpe artifacts/part1/bpe/ \
    --counts artifacts/part3/counts --output artifacts/part3/analysis
```

This writes `artifacts/part3/analysis/perplexity_by_language.tsv` and `perplexity_vs_n.svg`. It also writes `perplexity_vs_n.png` when `rsvg-convert` or Chrome is available. To redraw the chart from the table without scoring again:

```bash
uv run python part3_analysis/eval_languages.py \
    --tsv artifacts/part3/analysis/perplexity_by_language.tsv \
    --output artifacts/part3/analysis
```

`part3_analysis/` is a local helper for the writeup plot. It is not in the submission tree.

## Submission

Pack one zip with `pyproject.toml`, `uv.lock`, the scripts named in the assignment file tree, `part2_analysis/`, `writeup.pdf`, this README, and:

- `artifacts/part1/bpe/bpe_vocab.txt` and `bpe_merges.txt`
- `artifacts/part1/unigram/unigram.model`
- `artifacts/part2/model.pt`, `config.json`, and `tokenizer/` from the Unigram 50,000 run
- `artifacts/part3/model.json.gz`

Leave out `artifacts/part1/cleaned/`, the analysis directories, and the extra Part 2 files (`train.bin`, `dev.bin`, `dev.txt`, `dev_metrics.json`, `run_config.json`).
